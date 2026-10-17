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
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from voicemate.agent.tools.arxiv_tools import Paper, query_arxiv
from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.search import SearchBackend
from voicemate.memory.store import MemoryStore
from voicemate.memory.text import ttl_days_for

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


class ArxivApi:
    """The public arXiv API (through the ``arxiv`` library, run off the event loop)."""

    name = "arxiv"

    async def search(  # pragma: no cover - needs the network
        self, query: str, max_results: int
    ) -> list[Paper]:
        """Return up to ``max_results`` papers for ``query``, most relevant first."""
        return await asyncio.to_thread(query_arxiv, query, None, max_results)


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
    """

    def __init__(
        self,
        memory: MemoryStore,
        web_backend: SearchBackend,
        arxiv_backend: ArxivBackend,
        max_results: int,
    ) -> None:
        self.memory = memory
        self.web_backend = web_backend
        self.arxiv_backend = arxiv_backend
        self.max_results = max_results
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
        cached = await asyncio.to_thread(self.memory.cached_search, query, "web")
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
        await asyncio.to_thread(self.memory.store_search, query, "web", rows, ttl)
        for hit in hits:
            if hit.snippet:
                await asyncio.to_thread(
                    self.memory.add_documents,
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
        cached = await asyncio.to_thread(self.memory.cached_search, query, "arxiv")
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
        await asyncio.to_thread(self.memory.store_search, query, "arxiv", rows, ttl)
        for paper in papers:
            await asyncio.to_thread(
                self.memory.add_documents,
                [f"{paper.title}. {paper.summary}"],
                paper.abs_url,
                paper.title,
                "arxiv",
                ttl,
            )
        return Found(rows, False, query, self.arxiv_backend.name)
