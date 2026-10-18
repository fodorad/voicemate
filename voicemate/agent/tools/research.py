"""Research: cache-first searching and the rule for which addresses may be opened.

:class:`Research` is what the search tools stand on. Asking it for a web or arXiv search first
looks in long-term memory for a similar, still fresh search; only if there is none does it
query the backend, and then it stores what it found (as a reusable search and as documents for
semantic recall) with a freshness class that depends on the topic.

It also keeps the *listed addresses*: the URLs that searches returned (plus those the user
approved). ``fetch_page`` opens a listed address without asking and asks the user for any
other, because an address the model was talked into opening could carry private data in its
path or query.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from voicemate.agent.tools.arxiv_tools import Paper, normalize_id, query_arxiv
from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.documents import html_to_text, pdf_to_text
from voicemate.agent.tools.netguard import check_url, safe_get
from voicemate.agent.tools.search import SearchBackend
from voicemate.memory.store import MemoryStore
from voicemate.memory.text import chunk_text, ttl_days_for

logger = logging.getLogger(__name__)

#: Papers returned per arXiv search are limited to this range.
MIN_PAPERS, MAX_PAPERS = 1, 10


class ArxivBackend(Protocol):
    """A source of arXiv papers."""

    #: Short backend name.
    name: str

    async def search(self, query: str, max_results: int) -> list[Paper]:
        """Return up to ``max_results`` papers for ``query``."""
        ...

    async def get(self, paper_id: str) -> Paper | None:
        """The paper with this arXiv id, or None."""
        ...


class ArxivApi:
    """The public arXiv API (through the ``arxiv`` library, run off the event loop)."""

    name = "arxiv"

    async def search(  # pragma: no cover - needs the network
        self, query: str, max_results: int
    ) -> list[Paper]:
        """Return up to ``max_results`` papers for ``query``, most relevant first."""
        return await asyncio.to_thread(query_arxiv, query, None, max_results)

    async def get(self, paper_id: str) -> Paper | None:  # pragma: no cover - needs the network
        """The paper with this arXiv id, or None."""
        papers = await asyncio.to_thread(query_arxiv, None, [paper_id], 1)
        return papers[0] if papers else None


#: Asked before an unlisted address is opened. Returns None to allow it, or the user's own
#: words to refuse (they are passed on to the model, so "no, try X instead" still works).
Approval = Callable[[str], str | None]


@dataclass(frozen=True)
class Page:
    """The text of a web page or online PDF.

    Attributes:
        url: The address.
        text: The page's main text.
        from_memory: Read from memory; nothing left the machine.
        retrieved_at: Unix time the page was stored (from memory only).
    """

    url: str
    text: str
    from_memory: bool
    retrieved_at: float | None = None


@dataclass(frozen=True)
class Found:
    """The outcome of a cache-first search.

    Attributes:
        results: Result rows as stored (web hits or papers, as dictionaries).
        from_memory: Answered from a stored search; nothing left the machine.
        query: The query the results belong to (the stored one when from memory).
        backend: Name of the backend (when it was queried).
        retrieved_at: Unix time a stored search was made (from memory only).
    """

    results: list[dict[str, Any]]
    from_memory: bool
    query: str
    backend: str
    retrieved_at: float | None = None


class Research:
    """Cache-first search plus the set of addresses that may be opened without asking.

    Args:
        memory: Long-term memory holding stored searches and documents.
        web_backend: The web search engine.
        arxiv_backend: The arXiv source.
        max_results: Web results per search.
        http: HTTP client for pages and PDFs (every request passes the address guard).
        allowed_private_hosts: Hosts exempt from the address guard (tests only).
    """

    def __init__(
        self,
        memory: MemoryStore,
        web_backend: SearchBackend,
        arxiv_backend: ArxivBackend,
        max_results: int,
        http: httpx.AsyncClient,
        allowed_private_hosts: frozenset[str] = frozenset(),
    ) -> None:
        self.memory = memory
        self.web_backend = web_backend
        self.arxiv_backend = arxiv_backend
        self.max_results = max_results
        self.http = http
        self.allowed_private_hosts = allowed_private_hosts
        self._listed: set[str] = set()

    # ── listed addresses ──────────────────────────────────────────────────────

    def is_listed(self, url: str) -> bool:
        """True if a search returned (or the user approved) exactly this address."""
        return url in self._listed

    def list_address(self, url: str) -> None:
        """Remember that ``url`` may be opened without asking again."""
        self._listed.add(url)

    # ── searches ──────────────────────────────────────────────────────────────

    async def web(self, query: str) -> Found:
        """Search the web, answering from a fresh stored search when there is one.

        Raises:
            ToolError: If the backend fails (rate limit, network).
        """
        cached = await asyncio.to_thread(self.memory.research.cached_search, query, "web")
        if cached is not None:
            self._listed.update(r.get("url", "") for r in cached.results)
            return Found(
                cached.results, True, cached.query, self.web_backend.name, cached.retrieved_at
            )
        try:
            hits = await self.web_backend.search(query, self.max_results)
        except Exception as exc:  # network / rate limit: tell the model, don't crash the turn
            raise ToolError(f"Web search failed: {exc}") from exc
        rows = [asdict(h) for h in hits]
        self._listed.update(h.url for h in hits)
        ttl = ttl_days_for(query, "web")
        await asyncio.to_thread(self.memory.research.store_search, query, "web", rows, ttl)
        for hit in hits:
            if hit.snippet:
                await asyncio.to_thread(
                    self.memory.research.add_documents,
                    [f"{hit.title}: {hit.snippet}"],
                    hit.url,
                    hit.title,
                    "web",
                    ttl,
                )
        return Found(rows, False, query, self.web_backend.name)

    async def papers(self, query: str, max_results: int = 5) -> Found:
        """Search arXiv, answering from a fresh stored search when there is one.

        Args:
            query: Topic keywords.
            max_results: Number of papers, clamped to 1-10.

        Raises:
            ToolError: If arXiv cannot be reached.
        """
        cached = await asyncio.to_thread(self.memory.research.cached_search, query, "arxiv")
        if cached is not None:
            self._listed.update(
                f"https://arxiv.org/abs/{r.get('arxiv_id', '')}" for r in cached.results
            )
            return Found(
                cached.results, True, cached.query, self.arxiv_backend.name, cached.retrieved_at
            )
        count = max(MIN_PAPERS, min(MAX_PAPERS, int(max_results)))
        try:
            papers = await self.arxiv_backend.search(query, count)
        except Exception as exc:
            raise ToolError(f"arXiv search failed: {exc}") from exc
        rows = [asdict(p) for p in papers]
        self._listed.update(p.abs_url for p in papers)
        ttl = ttl_days_for(query, "arxiv")
        await asyncio.to_thread(self.memory.research.store_search, query, "arxiv", rows, ttl)
        for paper in papers:
            await asyncio.to_thread(
                self.memory.research.add_documents,
                [f"{paper.title}. {paper.summary}"],
                paper.abs_url,
                paper.title,
                "arxiv",
                ttl,
            )
        return Found(rows, False, query, self.arxiv_backend.name)

    # ── pages ─────────────────────────────────────────────────────────────────

    async def page(self, url: str, approve: Approval) -> Page:
        """The text of a page or online PDF: stored copy first, else fetched and stored.

        An address no search listed is opened only after ``approve`` allows it.

        Raises:
            ToolError: If the address is not a public http(s) address, the user refuses,
                the download fails, or the page has no readable text.
        """
        await check_url(url, self.allowed_private_hosts)
        stored = await asyncio.to_thread(self.memory.research.documents_for_url, url)
        if stored:
            text = "\n\n".join(hit.text for hit in stored)
            return Page(url, text, True, stored[0].created)
        if not self.is_listed(url):
            host = urlsplit(url).hostname or url
            refusal = approve(url)
            if refusal is not None:
                raise ToolError(f"The user did not approve opening {host}. They said: {refusal}")
            self.list_address(url)
        try:
            response = await safe_get(self.http, url, self.allowed_private_hosts)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Could not fetch {url}: {exc}") from exc
        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text, _ = await asyncio.to_thread(pdf_to_text, response.content)
            title = url.rsplit("/", 1)[-1]
        else:
            text = await asyncio.to_thread(html_to_text, response.text, url)
            title = url
        if not text.strip():
            raise ToolError("No readable text on that page")
        await asyncio.to_thread(
            self.memory.research.add_documents,
            chunk_text(text),
            url,
            title,
            "page",
            ttl_days_for(url, "page"),
        )
        return Page(url, text, False)

    # ── papers ────────────────────────────────────────────────────────────────

    async def get_paper(self, paper_id: str) -> Paper:
        """Look a paper up by arXiv id or URL.

        Raises:
            ToolError: If the id is not an arXiv id, arXiv fails, or the paper does not exist.
        """
        paper_id = normalize_id(paper_id)
        try:
            paper = await self.arxiv_backend.get(paper_id)
        except Exception as exc:
            raise ToolError(f"arXiv lookup failed: {exc}") from exc
        if paper is None:
            raise ToolError(f"No arXiv paper {paper_id}")
        return paper

    async def paper_pdf(self, paper: Paper) -> bytes:
        """Download the paper's PDF through the address guard."""
        response = await safe_get(self.http, paper.pdf_url, self.allowed_private_hosts)
        return response.content

    async def index_paper(self, paper: Paper, text: str) -> None:
        """Store the paper's full text for recall, once (the abstract is stored by searches)."""
        indexed = await asyncio.to_thread(self.memory.research.documents_for_url, paper.abs_url)
        if len(indexed) <= 1 and text:
            await asyncio.to_thread(
                self.memory.research.add_documents,
                chunk_text(text),
                paper.abs_url,
                paper.title,
                "paper",
                ttl_days_for("", "paper"),
            )
