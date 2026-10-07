"""Events flowing from the voice pipeline to the UI, and a small asyncio pub/sub bus.

The UI renders one *lane* per pipeline stage. Several lanes can be active at the same time
(the LLM keeps generating while earlier sentences are synthesized and spoken), so the status
model is per lane rather than a single exclusive state.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class Lane(StrEnum):
    """Pipeline stage shown as one status lane in the UI."""

    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    RECALLING = "recalling"
    THINKING = "thinking"
    TOOL = "tool"
    GENERATING = "generating"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"


class LaneState(StrEnum):
    """State of a lane."""

    IDLE = "idle"
    ACTIVE = "active"
    DONE = "done"


#: Where a tool got its answer from; ``web`` means data left the machine.
ToolSource = Literal["web", "memory", "local"]


def _now() -> float:
    return time.time()


@dataclass
class StatusEvent:
    """A lane changed state.

    Attributes:
        lane: The pipeline stage.
        state: Its new state.
        detail: Optional detail, e.g. the tool name.
        ts: Unix timestamp.
    """

    lane: Lane
    state: LaneState
    detail: str = ""
    ts: float = field(default_factory=_now)


@dataclass
class TranscriptEvent:
    """Recognized user speech (or typed text).

    Attributes:
        text: The transcript.
        final: False for live partial transcripts while the user is still speaking.
        lang: Detected language code (``"hu"`` or ``"en"``).
        merged: True when this text continues the previous, cut-off question and replaces it.
    """

    text: str
    final: bool
    lang: str = ""
    merged: bool = False


@dataclass
class TokenEvent:
    """A streamed piece of the assistant reply.

    Attributes:
        text: The new text.
    """

    text: str


@dataclass
class ReplyEvent:
    """The complete assistant reply of a turn.

    Attributes:
        text: Full reply text.
        interrupted: True when the user barged in before the reply finished.
    """

    text: str
    interrupted: bool = False


@dataclass
class ToolEvent:
    """A tool call started or finished.

    Attributes:
        name: Tool name.
        phase: ``"start"`` or ``"end"``.
        args: Tool arguments.
        source: Where the result came from (only meaningful at ``"end"``).
        summary: Short human-readable description (e.g. the query sent to the web).
    """

    name: str
    phase: Literal["start", "end"]
    args: dict[str, Any] = field(default_factory=dict)
    source: ToolSource = "local"
    summary: str = ""


@dataclass
class AudioEvent:
    """A chunk of synthesized speech for the browser to play.

    Attributes:
        pcm: Mono signed 16-bit little-endian PCM.
        sample_rate: Sample rate of ``pcm`` in Hz.
    """

    pcm: bytes
    sample_rate: int


@dataclass
class InterruptEvent:
    """The user barged in: stop playback immediately."""


@dataclass
class ErrorEvent:
    """Something failed in a way the user should know about.

    Attributes:
        message: Human-readable error.
    """

    message: str


@dataclass
class ReminderEvent:
    """A reminder became due.

    Attributes:
        note_id: Notes-store id.
        text: Reminder text.
    """

    note_id: int
    text: str


@dataclass
class ConfirmEvent:
    """The agent needs the user's permission (human-in-the-loop) or no longer does.

    Attributes:
        question: The question; empty when the confirmation was answered.
        open: True while waiting for the answer.
    """

    question: str
    open: bool = True


@dataclass
class HoldEvent:
    """The user started or ended "hold until I'm finished".

    Attributes:
        holding: True while pauses do not end the user's turn.
    """

    holding: bool


@dataclass
class NotesChangedEvent:
    """The notes store changed; the UI should refresh its notes panel."""


@dataclass
class MetricsEvent:
    """Latency metrics of a finished turn.

    Attributes:
        metrics: Mapping of metric name to value (seconds or counts).
    """

    metrics: dict[str, Any]


#: Any event published on the bus.
Event = (
    StatusEvent
    | TranscriptEvent
    | TokenEvent
    | ReplyEvent
    | ToolEvent
    | AudioEvent
    | InterruptEvent
    | ErrorEvent
    | ReminderEvent
    | ConfirmEvent
    | HoldEvent
    | NotesChangedEvent
    | MetricsEvent
)


class EventBus:
    """Fan-out pub/sub over asyncio queues.

    Publishing never blocks: when a subscriber falls behind, its oldest event is dropped.

    Args:
        maxsize: Per-subscriber queue capacity.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._queues: list[asyncio.Queue[Event]] = []

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._queues)

    def subscribe(self) -> asyncio.Queue[Event]:
        """Register a new subscriber queue."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._maxsize)
        self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Remove a subscriber queue (no-op if unknown)."""
        if queue in self._queues:
            self._queues.remove(queue)

    def publish(self, event: Event) -> None:
        """Deliver ``event`` to every subscriber without blocking."""
        for queue in self._queues:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    async def listen(self) -> AsyncIterator[Event]:
        """Iterate over events until the consuming task is cancelled."""
        queue = self.subscribe()
        try:
            while True:
                yield await queue.get()
        finally:
            self.unsubscribe(queue)
