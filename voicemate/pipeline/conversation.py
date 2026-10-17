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
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.types import Command

from voicemate.events import Event

logger = logging.getLogger(__name__)

#: Appended to a reply the user interrupted (kept in the conversation history).
INTERRUPTED_MARK: str = "[interrupted by the user]"

#: Result given to tool calls that were cancelled by a barge-in.
INTERRUPTED_TOOL_RESULT: str = "Cancelled: the user interrupted before this finished."


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

    def say(
        self, text: str, lang: str, message_id: str, emit: Callable[[Event], None]
    ) -> AsyncIterator[str]:
        """Add a user message and stream the assistant's answer text as it is written.

        Args:
            text: What the user said or typed.
            lang: Reply language of this turn.
            message_id: Identifies the user message (so it can be discarded later).
            emit: Receives the tool and status events of the run.

        Returns:
            An iterator of text chunks. When it ends, :meth:`pending_question` tells whether
            the agent paused to ask the user something.
        """
        graph_input = {"messages": [HumanMessage(text, id=message_id)], "lang": lang}
        return self._stream(graph_input, emit)

    def resume(self, answer: str, emit: Callable[[Event], None]) -> AsyncIterator[str]:
        """Answer the pending question and stream the rest of the run."""
        return self._stream(Command(resume=answer), emit)

    async def _stream(self, graph_input: Any, emit: Callable[[Event], None]) -> AsyncIterator[str]:
        run_config = {"configurable": {**self.run_config["configurable"], "emit": emit}}
        async for chunk, meta in self.graph.astream(
            graph_input, run_config, stream_mode="messages"
        ):
            if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                continue
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    async def close_interrupted(self, partial_reply: str) -> None:
        """Keep the history of a cut-off turn consistent and informative.

        Tool calls that never got a result are answered with a "cancelled" message (models
        reject dangling tool calls), and a partially spoken reply is kept, marked as
        interrupted, so the next turn knows what the user already heard. Safe to repeat.

        Args:
            partial_reply: The part of the answer the user heard (may be empty).
        """
        try:
            snapshot = await self.graph.aget_state(self.run_config)
        except Exception:  # pragma: no cover - checkpointer unavailable
            logger.exception("Cannot read the interrupted turn's state")
            return
        messages = snapshot.values.get("messages", []) if snapshot else []
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        patch: list = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            open_calls = [c for c in message.tool_calls if c["id"] not in answered]
            if open_calls:
                # Re-add the call message with its results: a cancel that lands just before
                # LangGraph commits the step can drop it, which would leave a tool result
                # without a call (models reject that). Messages are matched by id, so when
                # it is already stored this replaces it in place.
                patch.append(message)
                patch.extend(
                    ToolMessage(INTERRUPTED_TOOL_RESULT, tool_call_id=c["id"]) for c in open_calls
                )
        if partial_reply:
            patch.append(AIMessage(f"{partial_reply.strip()} {INTERRUPTED_MARK}"))
        if patch:
            await self.graph.aupdate_state(self.run_config, {"messages": patch}, as_node="agent")

    async def discard(self, message_id: str) -> None:
        """Remove an unanswered user message from the history (errors are ignored)."""
        try:
            await self.graph.aupdate_state(
                self.run_config, {"messages": [RemoveMessage(id=message_id)]}
            )
        except Exception:
            logger.debug("Could not discard message %s", message_id, exc_info=True)
