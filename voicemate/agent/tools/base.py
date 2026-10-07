"""Shared plumbing for agent tools.

Tools are plain async functions turned into LangChain tools. Each one receives the LangGraph
:class:`~langchain_core.runnables.RunnableConfig` (injected, invisible to the model) and uses
the ``emit`` callback stored under ``configurable`` to report :class:`ToolEvent` s, so the UI
can show whether data came from the web (left the machine), memory, or local state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voicemate.events import Event, ToolEvent, ToolSource

if TYPE_CHECKING:
    import httpx
    from langchain_core.runnables import RunnableConfig

    from voicemate.agent.tools.search import SearchBackend
    from voicemate.config import Config
    from voicemate.memory.notes import NotesStore
    from voicemate.memory.store import MemoryStore

#: HTTP User-Agent for outbound requests.
USER_AGENT: str = "voicemate/0.1 (+https://github.com/fodorad/voicemate; personal assistant)"


class ToolError(Exception):
    """A tool failed in a way the model should be told about (returned as the tool result)."""


@dataclass
class ToolContext:
    """Dependencies shared by all tools.

    Attributes:
        config: Application configuration.
        memory: Long-term memory store.
        notes: Notes database.
        search: Web search backend.
        http: Shared async HTTP client.
        now: Clock returning the current local time (injectable for tests).
        allowed_private_hosts: Hosts exempt from the SSRF guard (only for tests).
    """

    config: Config
    memory: MemoryStore
    notes: NotesStore
    search: SearchBackend
    http: httpx.AsyncClient
    now: Callable[[], datetime] = field(default=lambda: datetime.now().astimezone())
    allowed_private_hosts: frozenset[str] = frozenset()


def emitter(config: RunnableConfig | None) -> Callable[[Event], None]:
    """The event callback stored in ``config["configurable"]["emit"]`` (no-op if absent)."""
    configurable = (config or {}).get("configurable", {}) if config else {}
    emit = configurable.get("emit")
    return emit if callable(emit) else (lambda _event: None)


def report(
    config: RunnableConfig | None,
    name: str,
    args: dict[str, Any],
    source: ToolSource,
    summary: str = "",
) -> None:
    """Emit the ``end`` :class:`ToolEvent` of a tool call with its data source."""
    emitter(config)(ToolEvent(name, "end", args, source, summary))


def format_date(timestamp: float) -> str:
    """Unix timestamp → ``YYYY-MM-DD`` (local time)."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def display_path(path: Path) -> str:
    """Show paths under the home directory as ``~/...`` (keeps absolute paths out of replies)."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)
