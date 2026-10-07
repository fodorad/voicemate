"""Agent tools. :func:`build_tools` assembles every tool for one :class:`ToolContext`."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from voicemate.agent.tools import (
    arxiv_tools,
    clock_calc,
    fetch,
    files,
    memory_tools,
    notes_tools,
    search,
    weather,
)
from voicemate.agent.tools.base import ToolContext

#: Tools whose calls send data off the machine.
EGRESS_TOOLS: frozenset[str] = frozenset(
    {"web_search", "fetch_page", "arxiv_search", "download_paper", "get_weather"}
)


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    """All tools, in the order they are presented to the model."""
    modules = (clock_calc, weather, search, fetch, arxiv_tools, memory_tools, notes_tools, files)
    return [tool for module in modules for tool in module.build(ctx)]


__all__ = ["EGRESS_TOOLS", "ToolContext", "build_tools"]
