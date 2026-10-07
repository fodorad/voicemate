"""Explicit memory tools: store a durable fact, search everything remembered."""

from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.tools.base import ToolContext, format_date, report
from voicemate.memory.store import Recall


def format_recall(recall: Recall) -> str:
    """Render a :class:`Recall` with dates, grouped by kind."""
    if recall.is_empty:
        return "Nothing relevant in memory."
    lines: list[str] = []
    if recall.facts:
        lines.append("Known facts about the user:")
        lines += [f"- {h.text} (noted {format_date(h.created)})" for h in recall.facts]
    if recall.episodes:
        lines.append("Earlier conversations:")
        for h in recall.episodes:
            user = h.meta.get("user", "")
            assistant = str(h.meta.get("assistant", ""))[:300]
            lines.append(f"- {format_date(h.created)}: user asked {user!r}; answer: {assistant}")
    if recall.documents:
        lines.append("Research memory:")
        for h in recall.documents:
            source = h.meta.get("title") or h.meta.get("url", "")
            lines.append(
                f"- [{source}, retrieved {format_date(h.created)}, {h.meta.get('url', '')}] "
                f"{h.text[:600]}"
            )
    return "\n".join(lines)


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create ``remember`` and ``recall_memory``."""

    async def remember(fact: str, config: RunnableConfig) -> str:
        """Store a durable fact or preference about the user for future conversations.

        Examples: job goals, health conditions they mention, preferences, names of people.

        Args:
            fact: One self-contained sentence in English, e.g. "Adam prefers short answers".
        """
        replaced = await asyncio.to_thread(ctx.memory.add_fact, fact.strip())
        report(config, "remember", {"fact": fact}, "memory", fact)
        return "Updated an existing memory." if replaced else "Remembered."

    async def recall_memory(query: str, config: RunnableConfig) -> str:
        """Search long-term memory: user facts, earlier conversations and past research.

        Research covers web pages, search results and papers read before.

        Args:
            query: What to look for.
        """
        recall = await asyncio.to_thread(ctx.memory.recall, query)
        report(config, "recall_memory", {"query": query}, "memory", query)
        return format_recall(recall)

    return [
        StructuredTool.from_function(coroutine=remember, name="remember", parse_docstring=True),
        StructuredTool.from_function(
            coroutine=recall_memory, name="recall_memory", parse_docstring=True
        ),
    ]
