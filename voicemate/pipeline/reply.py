"""The reply of one turn: what is said, when, and how long it took.

A :class:`Reply` is created for each turn. The session feeds it the streamed answer text and the
agent's tool events; it decides what becomes speech (complete sentences, a spoken filler while
a network tool runs, an apology when the turn fails), remembers the answer text (so an
interrupted reply can be kept in the history) and the timings, and builds the turn's metrics.
Cancelling, the timeout, the speech worker and publishing events stay with the session.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from voicemate.agent.tools import EGRESS_TOOLS
from voicemate.asr.base import Transcript
from voicemate.pipeline.chunker import SpeechChunker

#: Spoken while a slow (network) tool runs and nothing has been said yet.
FILLERS: dict[str, str] = {"hu": "Egy pillanat, utánanézek.", "en": "One moment, let me check."}

#: Spoken when a turn fails or times out.
APOLOGIES: dict[str, str] = {
    "hu": "Elnézést, ez most nem sikerült. Kérlek, próbáld újra.",
    "en": "Sorry, that didn't work. Please try again.",
}


class Reply:
    """State and rules of one turn's answer.

    Args:
        lang: Reply language of the turn.
        ended_at: Monotonic time the user finished (reference for the timings).
        clock: Monotonic time source (injectable for tests).

    Attributes:
        text: The answer written so far.
        spoke: Something was queued for speech.
        failed: The turn failed.
        tools: Names of the tools the agent started.
        first_token: Monotonic time of the first token or tool start, if any.
        first_audio: Monotonic time the first audio was ready, if any.
        queue: Speech to synthesize, in order; ``None`` marks the end.
    """

    def __init__(self, lang: str, ended_at: float, clock: Callable[[], float] = time.monotonic):
        self.lang = lang
        self.ended_at = ended_at
        self.text = ""
        self.spoke = False
        self.failed = False
        self.tools: list[str] = []
        self.first_token: float | None = None
        self.first_audio: float | None = None
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._chunker = SpeechChunker()
        self._clock = clock

    def say(self, text: str) -> None:
        """Queue ``text`` for speech."""
        self.spoke = True
        self.queue.put_nowait(text)

    def mark_first_token(self) -> bool:
        """Record the first sign of progress; True only the first time."""
        if self.first_token is not None:
            return False
        self.first_token = self._clock()
        return True

    def audio_started(self) -> None:
        """Record when the first audio was ready (the first call counts)."""
        if self.first_audio is None:
            self.first_audio = self._clock()

    def add_text(self, chunk: str) -> bool:
        """Add streamed answer text; complete sentences are queued for speech.

        Returns:
            True when this is the first text of the answer.
        """
        first = not self.text
        self.text += chunk
        for piece in self._chunker.feed(chunk):
            self.say(piece)
        return first

    def on_tool_start(self, name: str) -> bool:
        """The agent started a tool: say a filler if it goes online and nothing was said yet.

        Returns:
            True when this was the first sign of progress.
        """
        self.tools.append(name)
        first = self.mark_first_token()
        if not self.spoke and name in EGRESS_TOOLS:
            self.say(FILLERS.get(self.lang, FILLERS["en"]))
        return first

    def fail(self, error: BaseException) -> str:
        """The turn failed: queue the apology.

        Returns:
            A short reason for the error message.
        """
        self.failed = True
        self.say(APOLOGIES.get(self.lang, APOLOGIES["en"]))
        if isinstance(error, TimeoutError):
            return "timed out"
        return str(error) or type(error).__name__

    def finish(self) -> None:
        """The answer is complete: queue what is left and close the queue."""
        for piece in self._chunker.flush():
            self.say(piece)
        self.queue.put_nowait(None)

    def metrics(self, transcript: Transcript | None) -> dict[str, Any]:
        """Timings and facts about this turn (seconds are relative to the user's last word)."""
        metrics: dict[str, Any] = {"lang": self.lang, "tools": self.tools}
        if transcript is not None:
            metrics["asr_s"] = round(transcript.latency_seconds, 3)
            metrics["audio_s"] = round(transcript.audio_seconds, 2)
        for key, value in (
            ("first_token_s", self.first_token),
            ("first_audio_s", self.first_audio),
        ):
            if value is not None:
                metrics[key] = round(value - self.ended_at, 3)
        metrics.update(
            total_s=round(self._clock() - self.ended_at, 3),
            chars_out=len(self.text),
            failed=self.failed,
            ts=datetime.now().isoformat(timespec="seconds"),
        )
        return metrics
