"""arXiv search and paper download.

Downloaded papers are saved as PDFs under ``[files] papers_dir`` (default
``~/Downloads/voicemate-papers``) so Adam can open them, and their text is indexed into
research memory so later questions are answered without re-downloading.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.tools.base import (
    ToolContext,
    ToolError,
    display_path,
    format_date,
    report,
)
from voicemate.agent.tools.documents import pdf_to_text
from voicemate.agent.tools.netguard import safe_get
from voicemate.memory.text import chunk_text, ttl_days_for

_ID = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")


@dataclass
class Paper:
    """arXiv metadata.

    Attributes:
        arxiv_id: Short id such as ``2401.01234v2``.
        title: Paper title.
        authors: Author names (first few).
        published: Publication date ``YYYY-MM-DD``.
        summary: Abstract.
        pdf_url: Direct PDF link.
    """

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    summary: str
    pdf_url: str

    @property
    def abs_url(self) -> str:
        """Abstract page URL."""
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def normalize_id(text: str) -> str:
    """Extract an arXiv id from free text or a URL.

    Raises:
        ToolError: If no id can be found.
    """
    match = _ID.search(text)
    if not match:
        raise ToolError(f"Not an arXiv id: {text!r}")
    return match.group(1)


def slugify(title: str, max_len: int = 60) -> str:
    """File-name-safe slug of a title."""
    slug = re.sub(r"[^\w]+", "-", title.lower(), flags=re.ASCII).strip("-")
    return slug[:max_len].rstrip("-") or "paper"


def paper_filename(paper: Paper) -> str:
    """``<id>-<slug>.pdf``."""
    return f"{paper.arxiv_id.replace('/', '_')}-{slugify(paper.title)}.pdf"


def _to_paper(result: Any) -> Paper:
    return Paper(
        arxiv_id=result.get_short_id(),
        title=" ".join(result.title.split()),
        authors=[a.name for a in result.authors[:5]],
        published=result.published.strftime("%Y-%m-%d"),
        summary=" ".join(result.summary.split()),
        pdf_url=result.pdf_url,
    )


def _query_arxiv(query: str | None, ids: list[str] | None, max_results: int) -> list[Paper]:
    import arxiv

    search = arxiv.Search(
        query=query or "",
        id_list=ids or [],
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_to_paper(r) for r in arxiv.Client(num_retries=2).results(search)]


def format_papers(papers: list[dict[str, Any]], header: str) -> str:
    """Numbered paper list for the model."""
    lines = [header]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        lines.append(
            f"{i}. [{p['arxiv_id']}] {p['title']} ({authors}, {p['published']})\n"
            f"   {p['summary'][:500]}"
        )
    if not papers:
        lines.append("No papers found.")
    return "\n".join(lines)


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create ``arxiv_search`` and ``download_paper``."""

    async def arxiv_search(query: str, config: RunnableConfig, max_results: int = 5) -> str:
        """Search arXiv for scientific papers (machine learning, physics, maths, ...).

        Args:
            query: Topic keywords, e.g. "mixture of experts routing".
            max_results: Number of papers (1-10).
        """
        cached = await asyncio.to_thread(ctx.memory.cached_search, query, "arxiv")
        if cached is not None:
            report(config, "arxiv_search", {"query": query}, "memory", cached.query)
            header = f"[from memory, retrieved {format_date(cached.retrieved_at)}]"
            return format_papers(cached.results, header)
        try:
            papers = await asyncio.to_thread(
                _query_arxiv, query, None, max(1, min(10, int(max_results)))
            )
        except Exception as exc:
            raise ToolError(f"arXiv search failed: {exc}") from exc
        rows = [asdict(p) for p in papers]
        ttl = ttl_days_for(query, "arxiv")
        await asyncio.to_thread(ctx.memory.store_search, query, "arxiv", rows, ttl)
        for paper in papers:
            await asyncio.to_thread(
                ctx.memory.add_documents,
                [f"{paper.title}. {paper.summary}"],
                paper.abs_url,
                paper.title,
                "arxiv",
                ttl,
            )
        report(config, "arxiv_search", {"query": query}, "web", query)
        return format_papers(rows, "[arXiv search]")

    async def download_paper(arxiv_id: str, config: RunnableConfig) -> str:
        """Download an arXiv paper as PDF and index its full text into memory.

        The PDF is saved in the papers folder so the user can open it; the indexed text lets
        you discuss and summarize it later.

        Args:
            arxiv_id: arXiv id or URL, e.g. "2401.01234".
        """
        paper_id = normalize_id(arxiv_id)
        try:
            papers = await asyncio.to_thread(_query_arxiv, None, [paper_id], 1)
        except Exception as exc:
            raise ToolError(f"arXiv lookup failed: {exc}") from exc
        if not papers:
            raise ToolError(f"No arXiv paper {paper_id}")
        paper = papers[0]
        folder = ctx.config.files.papers_path
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / paper_filename(paper)
        if not path.exists():
            response = await safe_get(ctx.http, paper.pdf_url, ctx.allowed_private_hosts)
            path.write_bytes(response.content)
        text, pages = await asyncio.to_thread(pdf_to_text, path.read_bytes())
        indexed = await asyncio.to_thread(ctx.memory.documents_for_url, paper.abs_url)
        # Only the abstract is stored by arxiv_search; index the full text once.
        if len(indexed) <= 1 and text:
            await asyncio.to_thread(
                ctx.memory.add_documents,
                chunk_text(text),
                paper.abs_url,
                paper.title,
                "paper",
                ttl_days_for("", "paper"),
            )
        report(config, "download_paper", {"arxiv_id": paper_id}, "web", paper.title)
        return (
            f"Saved '{paper.title}' ({pages} pages) to {display_path(path)} and indexed it "
            f"into memory.\nAbstract: {paper.summary[:800]}"
        )

    return [
        StructuredTool.from_function(
            coroutine=arxiv_search, name="arxiv_search", parse_docstring=True
        ),
        StructuredTool.from_function(
            coroutine=download_paper, name="download_paper", parse_docstring=True
        ),
    ]
