"""Full-duplex voice session: listen, transcribe, think, speak and get interrupted.

One :class:`VoiceSession` exists per browser tab. Microphone PCM arrives via
:meth:`VoiceSession.feed_audio`; everything the UI needs, including synthesized audio,
leaves through :attr:`VoiceSession.bus`.

Concurrency: the VAD runs inline (well under 1 ms per frame); ASR and TTS run on the
worker threads inside :class:`~voicemate.speech.Speech`; each turn is one asyncio task in
which the LLM stream feeds a sentence chunker whose output is synthesized by a parallel TTS
worker, so speech starts while the model is still generating. At most one turn runs at a
time: speaking (or typing) while a turn is active cancels it (barge-in) before the new one
starts.

"Hold until I'm finished" (:meth:`VoiceSession.hold`) suspends endpointing: utterances are
transcribed and collected, and :meth:`VoiceSession.release` sends them as one question, so
the user can pause to think without the assistant jumping in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from voicemate.agent.hitl import Confirmation
from voicemate.agent.tools import EGRESS_TOOLS
from voicemate.asr.base import Transcript
from voicemate.audio.codec import SAMPLE_RATE, float32_to_pcm16, pcm16_to_float32
from voicemate.audio.vad import Segmenter, SpeechEnd, SpeechStart
from voicemate.events import (
    AudioEvent,
    ConfirmEvent,
    ErrorEvent,
    Event,
    EventBus,
    HoldEvent,
    InterruptEvent,
    Lane,
    LaneState,
    MetricsEvent,
    ReminderEvent,
    ReplyEvent,
    StatusEvent,
    TokenEvent,
    ToolEvent,
    TranscriptEvent,
)
from voicemate.pipeline.chunker import SpeechChunker
from voicemate.pipeline.conversation import (
    ChatLine,
    Conversation,
)
from voicemate.runtime import Runtime

logger = logging.getLogger(__name__)

#: Spoken while a slow (network) tool runs and nothing has been said yet.
FILLERS: dict[str, str] = {"hu": "Egy pillanat, utánanézek.", "en": "One moment, let me check."}

#: Spoken when a turn fails or times out.
APOLOGIES: dict[str, str] = {
    "hu": "Elnézést, ez most nem sikerült. Kérlek, próbáld újra.",
    "en": "Sorry, that didn't work. Please try again.",
}

#: A question cut off before any answer is merged with the next utterance only if that
#: utterance follows within this many seconds; later speech is a new question.
MERGE_WINDOW_S: float = 20.0

#: Live partial transcripts only cover the most recent speech, so a long utterance never
#: makes the final transcript wait behind a slow partial pass on the shared ASR thread.
PARTIAL_WINDOW_S: float = 8.0

#: Seconds between checks for due reminders.
REMINDER_INTERVAL_S: float = 30.0


@dataclass
class _Pending:
    """A user message whose turn was cancelled before anything was said."""

    message_id: str
    text: str
    at: float


@dataclass
class _Turn:
    """Mutable state of one turn while it streams."""

    lang: str
    ended_at: float
    reply: str = ""
    spoke: bool = False
    first_token: float | None = None
    first_audio: float | None = None
    failed: bool = False
    tools: list[str] = field(default_factory=list)
    chunker: SpeechChunker = field(default_factory=SpeechChunker)
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)

    def say(self, text: str) -> None:
        """Queue ``text`` for speech."""
        self.spoke = True
        self.queue.put_nowait(text)


class VoiceSession:
    """One conversation with one browser tab.

    Args:
        runtime: Shared models and stores.
        thread_id: Conversation id for the checkpointer (a new one per session by default).
    """

    def __init__(self, runtime: Runtime, thread_id: str | None = None) -> None:
        self.runtime = runtime
        self.config = runtime.config
        self.conversation = Conversation(
            runtime.graph, thread_id or f"session-{uuid.uuid4().hex[:12]}"
        )
        self.bus = EventBus()
        self.segmenter = Segmenter(runtime.vad_factory(), self.config.vad)
        self.lang = self.config.assistant.default_language
        #: While True, utterances are collected instead of answered (see :meth:`hold`).
        self.holding = False
        self._held: list[str] = []
        self._utterances: set[asyncio.Task] = set()
        #: Question the agent is waiting on (human-in-the-loop), if any.
        self.confirmation: Confirmation | None = None
        self._speaking = False
        self._turn: asyncio.Task | None = None
        self._partial: asyncio.Task | None = None
        self._reminders: asyncio.Task | None = None
        self._last_partial_at = 0.0
        self._pending: _Pending | None = None
        self._interrupted_reply = ""  # partial reply of the turn cut off by the last barge-in
        self._tasks: set[asyncio.Task] = set()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background duties (reminder watcher)."""
        if self.runtime.notes is not None and self._reminders is None:
            self._reminders = self._spawn(self._watch_reminders())

    async def close(self) -> None:
        """Cancel everything this session is doing."""
        for task in list(self._tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def wait_idle(self) -> None:
        """Wait until no utterance, turn or follow-up task is in progress."""
        while pending := [t for t in self._tasks if t is not self._reminders and not t.done()]:
            await asyncio.wait(pending)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Start a task that is kept alive until done and whose errors are logged.

        The event loop holds only weak references to tasks, so an untracked task can be
        garbage-collected while it runs.
        """
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Session task failed", exc_info=task.exception())

    @property
    def thread_id(self) -> str:
        """Id of this session's conversation thread."""
        return self.conversation.thread_id

    @property
    def busy(self) -> bool:
        """True while a turn is being processed or its audio is playing."""
        return self._speaking or (self._turn is not None and not self._turn.done())

    def emit(self, event: Event) -> None:
        """Publish an event; tool events also drive the TOOL lane."""
        if isinstance(event, ToolEvent):
            state = LaneState.ACTIVE if event.phase == "start" else LaneState.DONE
            self.bus.publish(StatusEvent(Lane.TOOL, state, event.name))
        self.bus.publish(event)

    def _status(self, lane: Lane, state: LaneState, detail: str = "") -> None:
        self.bus.publish(StatusEvent(lane, state, detail))

    # ── inputs from the browser ──────────────────────────────────────────────

    async def feed_audio(self, pcm: bytes) -> None:
        """Consume one frame of 16 kHz mono PCM16 microphone audio."""
        for event in self.segmenter.feed(pcm16_to_float32(pcm)):
            if isinstance(event, SpeechStart):
                self._status(Lane.LISTENING, LaneState.ACTIVE)
                if self.busy:
                    await self.interrupt()
            elif isinstance(event, SpeechEnd):
                self._utterance_ended(event)
        self._maybe_partial()

    def _utterance_ended(self, event: SpeechEnd) -> None:
        self._status(Lane.LISTENING, LaneState.DONE)
        self._last_partial_at = 0.0
        task = self._spawn(self._handle_utterance(event.audio))
        self._utterances.add(task)
        task.add_done_callback(self._utterances.discard)

    async def submit_text(self, text: str) -> None:
        """Handle typed input like a spoken utterance (no ASR, never merged)."""
        text = text.strip()
        if text and self.holding:
            self._collect(text)
        elif text:
            await self._begin_turn(text, time.monotonic(), continuation=False)

    async def answer_confirmation(self, answer: str) -> None:
        """Answer the pending human-in-the-loop question (e.g. from a UI button)."""
        if self.confirmation is not None:
            await self._begin_turn(answer, time.monotonic(), continuation=False)

    async def interrupt(self) -> None:
        """Barge-in: stop the current turn and tell the browser to stop playback."""
        self.bus.publish(InterruptEvent())
        self._speaking = False
        self.segmenter.strict = False
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn
            await self._close_interrupted_turn()
        for lane in (Lane.THINKING, Lane.TOOL, Lane.GENERATING, Lane.SYNTHESIZING, Lane.SPEAKING):
            self._status(lane, LaneState.IDLE)

    def playback_started(self) -> None:
        """The browser started playing assistant audio."""
        self._speaking = True
        self.segmenter.strict = True
        self._status(Lane.SPEAKING, LaneState.ACTIVE)

    def playback_ended(self) -> None:
        """The browser finished playing all queued audio."""
        self._speaking = False
        self.segmenter.strict = False
        self._status(Lane.SPEAKING, LaneState.DONE)

    def microphone_stopped(self) -> None:
        """The user turned the microphone off: answer what was heard so far.

        A sentence still in progress is ended here, and a hold is released, because turning
        the microphone off means "I'm done talking".
        """
        end = self.segmenter.flush()
        if end is not None:
            self._utterance_ended(end)
        else:
            self._status(Lane.LISTENING, LaneState.IDLE)
        if self.holding:
            self._spawn(self.release())

    async def hold(self) -> None:
        """Keep listening until :meth:`release`; pauses no longer end the user's turn.

        A running answer is stopped first: the user wants the floor.
        """
        if self.busy:
            await self.interrupt()
        self.holding = True
        self.bus.publish(HoldEvent(True))

    async def release(self) -> None:
        """The user is finished: send everything said (or typed) during the hold."""
        if not self.holding:
            return
        if (end := self.segmenter.flush()) is not None:
            self._utterance_ended(end)
        while self._utterances:  # transcripts still on their way join the question
            await asyncio.wait(set(self._utterances))
        self.holding = False
        self.bus.publish(HoldEvent(False))
        held, self._held = self._held, []
        if held:
            await self._begin_turn(" ".join(held), time.monotonic(), shown=True)

    def _collect(self, text: str) -> None:
        """Add one held utterance; the UI shows the question growing in one bubble."""
        self._held.append(text)
        self.lang = self.runtime.langid.detect(text, previous=self.lang)
        joined = " ".join(self._held)
        self.bus.publish(
            TranscriptEvent(joined, final=True, lang=self.lang, merged=len(self._held) > 1)
        )

    async def restore(self) -> list[ChatLine]:
        """Pick up this thread where it stopped (a reopened page).

        Reopens a question the agent is still waiting on and returns what was said so far.
        """
        if question := await self.conversation.pending_question():
            self.confirmation = Confirmation(question)
            self.bus.publish(ConfirmEvent(question))
        return await self.conversation.lines()

    # ── ASR ──────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        speech = self.runtime.speech
        return "" if speech is None else await speech.transcribe(audio)

    def _maybe_partial(self) -> None:
        """Schedule a live partial transcript every ``partial_interval_s`` of speech."""
        if not self.segmenter.in_speech or (self._partial and not self._partial.done()):
            return
        seconds = self.segmenter.speech_seconds
        if seconds - self._last_partial_at < self.config.asr.partial_interval_s:
            return
        self._last_partial_at = seconds
        audio = self.segmenter.current_audio
        window = int(PARTIAL_WINDOW_S * SAMPLE_RATE)
        self._partial = self._spawn(self._publish_partial(audio[-window:], len(audio) > window))

    async def _publish_partial(self, audio: np.ndarray, truncated: bool) -> None:
        text = await self._transcribe(audio)
        if text and self.segmenter.in_speech:
            self.bus.publish(TranscriptEvent(f"… {text}" if truncated else text, final=False))

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        ended_at = time.monotonic()
        self._status(Lane.TRANSCRIBING, LaneState.ACTIVE)
        try:
            text = await self._transcribe(audio)
        except Exception as exc:
            logger.exception("Speech recognition failed")
            self._status(Lane.TRANSCRIBING, LaneState.IDLE)
            self.bus.publish(ErrorEvent(f"Speech recognition failed: {exc}"))
            return
        latency = time.monotonic() - ended_at
        self._status(Lane.TRANSCRIBING, LaneState.DONE)
        if self.config.save_audio:
            await asyncio.to_thread(self._save_audio, audio)
        if not any(ch.isalnum() for ch in text):
            return
        if self.holding:
            self._collect(text.strip())
            return
        transcript = Transcript(text, "", len(audio) / SAMPLE_RATE, latency)
        await self._begin_turn(text, ended_at, transcript, continuation=True)

    def _save_audio(self, audio: np.ndarray) -> None:
        import soundfile as sf

        folder = self.config.data_path / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}.wav", audio, SAMPLE_RATE)

    # ── turns ────────────────────────────────────────────────────────────────

    async def _begin_turn(
        self,
        text: str,
        ended_at: float,
        transcript: Transcript | None = None,
        continuation: bool = False,
        shown: bool = False,
    ) -> None:
        """The single entry point for new user input: at most one turn runs at a time.

        Args:
            text: What the user said or typed.
            ended_at: Monotonic time the user finished (latency reference).
            transcript: ASR details when the input was spoken.
            continuation: Spoken input may continue a question cut off moments ago.
            shown: The UI already shows this text (collected during a hold).
        """
        if self._turn is not None and not self._turn.done():
            await self.interrupt()
        resume = None
        if self.confirmation is not None:  # the user's words answer the pending question
            resume, self.confirmation = text, None
            self.bus.publish(ConfirmEvent("", open=False))
        pending, self._pending = self._pending, None
        merged = bool(
            resume is None and continuation and pending and ended_at - pending.at <= MERGE_WINDOW_S
        )
        if merged and pending is not None:
            await self.conversation.discard(pending.message_id)
            text = f"{pending.text} {text}"
        lang = self.runtime.langid.detect(text, previous=self.lang)
        self.lang = lang
        self.bus.publish(TranscriptEvent(text, final=True, lang=lang, merged=merged or shown))
        self._turn = self._spawn(self._run_turn(text, lang, ended_at, transcript, resume))

    async def _close_interrupted_turn(self) -> None:
        """Repair the history of a cut-off turn (see :meth:`Conversation.close_interrupted`)."""
        partial, self._interrupted_reply = self._interrupted_reply, ""
        await self.conversation.close_interrupted(partial)

    async def _run_turn(
        self,
        text: str,
        lang: str,
        ended_at: float,
        transcript: Transcript | None,
        resume: str | None = None,
    ) -> None:
        """Stream one agent run into speech; ``resume`` answers a pending confirmation."""
        turn = _Turn(lang, ended_at)
        message_id = uuid.uuid4().hex
        emit = self._tool_emitter(turn)
        chunks = (
            self.conversation.resume(resume, emit)
            if resume is not None
            else self.conversation.say(text, lang, message_id, emit)
        )
        worker = asyncio.create_task(self._tts_worker(turn))
        self._status(Lane.THINKING, LaneState.ACTIVE)
        try:
            async with asyncio.timeout(self.config.llm.turn_timeout_s):
                await self._stream(turn, chunks)
                await self._ask_if_paused(turn)
        except asyncio.CancelledError:
            worker.cancel()
            if turn.reply:
                self._interrupted_reply = turn.reply
                self.bus.publish(ReplyEvent(turn.reply, interrupted=True))
            elif resume is None:
                self._pending = _Pending(message_id, text, time.monotonic())
            raise
        except Exception as exc:  # timeout, Ollama down, ...
            turn.failed = True
            logger.exception("Turn failed")
            reason = (
                "timed out" if isinstance(exc, TimeoutError) else str(exc) or type(exc).__name__
            )
            self.bus.publish(ErrorEvent(f"Turn failed: {reason}"))
            turn.say(APOLOGIES.get(lang, APOLOGIES["en"]))
        self._status(Lane.THINKING, LaneState.IDLE)
        for piece in turn.chunker.flush():
            turn.say(piece)
        turn.queue.put_nowait(None)
        self._status(Lane.GENERATING, LaneState.DONE)
        await worker
        self.bus.publish(ReplyEvent(turn.reply))
        self._finish_metrics(turn, transcript)
        if turn.reply and self.runtime.memory is not None and not turn.failed:
            memory = self.runtime.memory
            self._spawn(
                asyncio.to_thread(memory.add_episode, text, turn.reply, lang, self.thread_id)
            )

    def _tool_emitter(self, turn: _Turn):
        """The ``emit`` callback for one turn: publishes events and speaks a filler."""

        def on_event(event: Event) -> None:
            self.emit(event)
            if not (isinstance(event, ToolEvent) and event.phase == "start"):
                return
            turn.tools.append(event.name)
            if turn.first_token is None:
                turn.first_token = time.monotonic()
                self._status(Lane.THINKING, LaneState.DONE)
            if not turn.spoke and event.name in EGRESS_TOOLS:
                turn.say(FILLERS.get(turn.lang, FILLERS["en"]))

        return on_event

    async def _stream(self, turn: _Turn, chunks: AsyncIterator[str]) -> None:
        """Feed the agent's streamed answer into the chunker and the UI."""
        async for content in chunks:
            if turn.first_token is None:
                turn.first_token = time.monotonic()
                self._status(Lane.THINKING, LaneState.DONE)
            if not turn.reply:
                self._status(Lane.GENERATING, LaneState.ACTIVE)
            turn.reply += content
            self.bus.publish(TokenEvent(content))
            for piece in turn.chunker.feed(content):
                turn.say(piece)

    async def _ask_if_paused(self, turn: _Turn) -> None:
        """If a tool paused the graph for permission, show and speak the question."""
        question = await self.conversation.pending_question()
        if question is None:
            return
        self.confirmation = Confirmation(question)
        self.bus.publish(ConfirmEvent(question))
        turn.say(question)

    async def _tts_worker(self, turn: _Turn) -> None:
        while (piece := await turn.queue.get()) is not None:
            if self.runtime.speech is None:
                continue
            self._status(Lane.SYNTHESIZING, LaneState.ACTIVE)
            if await self._speak(piece, turn.lang) and turn.first_audio is None:
                turn.first_audio = time.monotonic()
        self._status(Lane.SYNTHESIZING, LaneState.DONE)

    async def _speak(self, text: str, lang: str) -> bool:
        """Synthesize ``text`` and publish it as audio; False when there was nothing to play."""
        speech = self.runtime.speech
        if speech is None:
            return False
        audio, rate = await speech.synthesize(text, lang)
        if not len(audio):
            return False
        self.bus.publish(AudioEvent(float32_to_pcm16(audio), rate))
        return True

    def _finish_metrics(self, turn: _Turn, transcript: Transcript | None) -> None:
        metrics: dict[str, Any] = {"lang": turn.lang, "tools": turn.tools}
        if transcript is not None:
            metrics["asr_s"] = round(transcript.latency_seconds, 3)
            metrics["audio_s"] = round(transcript.audio_seconds, 2)
        for key, value in (
            ("first_token_s", turn.first_token),
            ("first_audio_s", turn.first_audio),
        ):
            if value is not None:
                metrics[key] = round(value - turn.ended_at, 3)
        metrics.update(
            total_s=round(time.monotonic() - turn.ended_at, 3),
            chars_out=len(turn.reply),
            failed=turn.failed,
            ts=datetime.now().isoformat(timespec="seconds"),
        )
        self.bus.publish(MetricsEvent(metrics))
        log = turns_log_path(self.config.data_path)
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except OSError:  # pragma: no cover - disk problems must not break the conversation
            logger.warning("Could not write %s", log)

    # ── reminders ────────────────────────────────────────────────────────────

    async def check_reminders(self) -> int:
        """Announce due reminders once; returns how many were announced.

        A reminder is always shown in the UI; it is also spoken when the assistant is idle.
        """
        notes = self.runtime.notes
        if notes is None:
            return 0
        due = await asyncio.to_thread(notes.due_reminders)
        for note in due:
            await asyncio.to_thread(notes.mark_notified, note.id)
            self.bus.publish(ReminderEvent(note.id, note.text))
            if not self.busy and not self.holding:
                prefix = "Emlékeztető: " if self.lang == "hu" else "Reminder: "
                await self._speak(prefix + note.text, self.lang)
        return len(due)

    async def _watch_reminders(self) -> None:
        while True:
            try:
                await self.check_reminders()
            except Exception:  # pragma: no cover - keep watching whatever happens
                logger.exception("Reminder check failed")
            await asyncio.sleep(REMINDER_INTERVAL_S)


def turns_log_path(data_path: Path) -> Path:
    """Location of the per-turn metrics log."""
    return data_path / "logs" / "turns.jsonl"
