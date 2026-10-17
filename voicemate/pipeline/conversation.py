"""The conversation: one persisted thread of messages between the user and the assistant.

A conversation is a LangGraph thread. This module owns everything about its identity and
history so the voice session and the UI never touch the checkpointer: which thread is
current (and that it survives page reloads and restarts), what was said so far, and which
question, if any, the agent is waiting on. It is not long-term memory (see
:mod:`voicemate.memory`): starting a new conversation keeps facts, research and notes.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

#: Appended to a reply the user interrupted (kept in the conversation history).
INTERRUPTED_MARK: str = "[interrupted by the user]"


@dataclass
class ChatLine:
    """One message of the conversation as the user saw it (for re-rendering a page).

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        text: What was said, without internal markers.
        interrupted: The user cut this reply off.
    """

    role: str
    text: str
    interrupted: bool = False


def question_text(value: Any) -> str:
    """The question text of a LangGraph ``interrupt()`` payload."""
    return str(value.get("question", value) if isinstance(value, dict) else value)


class ThreadStore:
    """Remembers which conversation is current, across page reloads and restarts.

    Args:
        path: JSON file holding the current thread id.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def current(self) -> str:
        """The current thread id; a new conversation is started when none is stored."""
        try:
            return str(json.loads(self.path.read_text())["thread_id"])
        except (OSError, ValueError, KeyError, TypeError):
            return self.start_new()

    def start_new(self) -> str:
        """Begin a fresh conversation and make it the current one."""
        thread = f"conversation-{uuid.uuid4().hex[:12]}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"thread_id": thread}))
        return thread


class Conversation:
    """One thread of the agent graph: its history and its pending question.

    Args:
        graph: The compiled LangGraph agent (with a checkpointer).
        thread_id: Identifies the thread inside the checkpointer.
    """

    def __init__(self, graph: Any, thread_id: str) -> None:
        self.graph = graph
        self.thread_id = thread_id

    @property
    def run_config(self) -> dict[str, Any]:
        """LangGraph run configuration selecting this thread."""
        return {"configurable": {"thread_id": self.thread_id}}

    async def lines(self) -> list[ChatLine]:
        """What was said so far, in order (tool traffic is not shown)."""
        snapshot = await self.graph.aget_state(self.run_config)
        lines: list[ChatLine] = []
        for message in snapshot.values.get("messages", []):
            text = str(message.content).strip()
            if isinstance(message, HumanMessage) and text:
                lines.append(ChatLine("user", text))
            elif isinstance(message, AIMessage) and text:
                interrupted = text.endswith(INTERRUPTED_MARK)
                lines.append(
                    ChatLine("assistant", text.removesuffix(INTERRUPTED_MARK).strip(), interrupted)
                )
        return lines

    async def pending_question(self) -> str | None:
        """The question the agent is waiting on (human-in-the-loop), if any."""
        snapshot = await self.graph.aget_state(self.run_config)
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        return question_text(interrupts[0].value) if interrupts else None
