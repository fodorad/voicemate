"""Full-duplex voice session: listen, transcribe, think, speak and get interrupted.

One :class:`VoiceSession` exists per browser tab. Microphone PCM arrives via
:meth:`VoiceSession.feed_audio`; everything the UI needs, including synthesized audio,
leaves through :attr:`VoiceSession.bus`.

Concurrency: the VAD runs inline (well under 1 ms per frame); ASR and TTS run on their
single-thread executors; each turn is one asyncio task in which the LLM stream feeds a
sentence chunker whose output is synthesized by a parallel TTS worker, so speech starts
while the model is still generating. At most one turn runs at a time: speaking (or typing)
while a turn is active cancels it (barge-in) before the new one starts.

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
from collections.abc import Coroutine
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
from voicemate.runtime import Runtime

logger = logging.getLogger(__name__)

#: Spoken while a slow (network) tool runs and nothing has been said yet.
FILLERS: dict[str, str] = {"hu": "Egy pillanat, utánanézek.", "en": "One moment, let me check."}

#: Spoken when a turn fails or times out.
APOLOGIES: dict[str, str] = {
    "hu": "Elnézést, ez most nem sikerült. Kérlek, próbáld újra.",
    "en": "Sorry, that didn't work. Please try again.",
}

#: Appended to a reply the user interrupted (kept in the conversation history).
INTERRUPTED_MARK: str = "[interrupted by the user]"

#: Result given to tool calls that were cancelled by a barge-in.
INTERRUPTED_TOOL_RESULT: str = "Cancelled: the user interrupted before this finished."

#: A question cut off before any answer is merged with the next utterance only if that
#: utterance follows within this many seconds; later speech is a new question.
MERGE_WINDOW_S: float = 20.0

#: Live partial transcripts only cover the most recent speech, so a long utterance never
#: makes the final transcript wait behind a slow partial pass on the shared ASR thread.
PARTIAL_WINDOW_S: float = 8.0

#: Seconds between checks for due reminders.
REMINDER_INTERVAL_S: float = 30.0


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
        self.thread_id = thread_id or f"session-{uuid.uuid4().hex[:12]}"
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
        from langchain_core.messages import AIMessage, HumanMessage

        snapshot = await self.runtime.graph.aget_state(
            {"configurable": {"thread_id": self.thread_id}}
        )
        lines: list[ChatLine] = []
        for message in snapshot.values.get("messages", []):
            text = str(message.content).strip()
            if isinstance(message, HumanMessage) and text:
                lines.append(ChatLine("user", text))
            elif isinstance(message, AIMessage) and text:
                interrupted = text.endswith(INTERRUPTED_MARK)
                text = text.removesuffix(INTERRUPTED_MARK).strip()
                lines.append(ChatLine("assistant", text, interrupted))
        if snapshot.interrupts:
            question = self._question(snapshot.interrupts[0].value)
            self.confirmation = Confirmation(question)
            self.bus.publish(ConfirmEvent(question))
        return lines

    # ── ASR ──────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        recognizer = self.runtime.recognizer
        if recognizer is None:
            return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.runtime.asr_executor, recognizer.transcribe_text, audio
        )

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
            await self._discard_message(pending.message_id)
            text = f"{pending.text} {text}"
        lang = self.runtime.langid.detect(text, previous=self.lang)
        self.lang = lang
        self.bus.publish(TranscriptEvent(text, final=True, lang=lang, merged=merged or shown))
        self._turn = self._spawn(self._run_turn(text, lang, ended_at, transcript, resume))

    async def _close_interrupted_turn(self) -> None:
        """Keep the history of a cut-off turn consistent and informative.

        Tool calls that never got a result are answered with a "cancelled" message (models
        reject dangling tool calls), and a partially spoken reply is kept, marked as
        interrupted, so the next turn knows what the user already heard.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        partial, self._interrupted_reply = self._interrupted_reply, ""
        run_config = {"configurable": {"thread_id": self.thread_id}}
        try:
            snapshot = await self.runtime.graph.aget_state(run_config)
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
        if partial:
            patch.append(AIMessage(f"{partial.strip()} {INTERRUPTED_MARK}"))
        if patch:
            await self.runtime.graph.aupdate_state(run_config, {"messages": patch}, as_node="agent")

    async def _discard_message(self, message_id: str) -> None:
        """Remove an unanswered user message from the checkpointed history."""
        from langchain_core.messages import RemoveMessage

        with contextlib.suppress(Exception):
            await self.runtime.graph.aupdate_state(
                {"configurable": {"thread_id": self.thread_id}},
                {"messages": [RemoveMessage(id=message_id)]},
            )

    async def _run_turn(
        self,
        text: str,
        lang: str,
        ended_at: float,
        transcript: Transcript | None,
        resume: str | None = None,
    ) -> None:
        """Stream one agent run into speech; ``resume`` answers a pending confirmation."""
        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        turn = _Turn(lang, ended_at)
        message = HumanMessage(text, id=uuid.uuid4().hex)
        graph_input: Any = (
            Command(resume=resume) if resume is not None else {"messages": [message], "lang": lang}
        )
        run_config = {
            "configurable": {"thread_id": self.thread_id, "emit": self._tool_emitter(turn)}
        }
        worker = asyncio.create_task(self._tts_worker(turn))
        self._status(Lane.THINKING, LaneState.ACTIVE)
        try:
            async with asyncio.timeout(self.config.llm.turn_timeout_s):
                await self._stream(turn, graph_input, run_config)
                await self._ask_if_paused(turn, run_config)
        except asyncio.CancelledError:
            worker.cancel()
            if turn.reply:
                self._interrupted_reply = turn.reply
                self.bus.publish(ReplyEvent(turn.reply, interrupted=True))
            elif resume is None:
                self._pending = _Pending(message.id or "", text, time.monotonic())
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

    async def _stream(self, turn: _Turn, graph_input: Any, run_config: dict) -> None:
        """Feed the agent's streamed answer into the chunker and the UI."""
        from langchain_core.messages import AIMessageChunk

        async for chunk, meta in self.runtime.graph.astream(
            graph_input, run_config, stream_mode="messages"
        ):
            if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                continue
            content = chunk.content
            if not isinstance(content, str) or not content:
                continue
            if turn.first_token is None:
                turn.first_token = time.monotonic()
                self._status(Lane.THINKING, LaneState.DONE)
            if not turn.reply:
                self._status(Lane.GENERATING, LaneState.ACTIVE)
            turn.reply += content
            self.bus.publish(TokenEvent(content))
            for piece in turn.chunker.feed(content):
                turn.say(piece)

    async def _ask_if_paused(self, turn: _Turn, run_config: dict) -> None:
        """If a tool paused the graph for permission, show and speak the question."""
        snapshot = await self.runtime.graph.aget_state(run_config)
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if not interrupts:
            return
        question = self._question(interrupts[0].value)
        self.confirmation = Confirmation(question)
        self.bus.publish(ConfirmEvent(question))
        turn.say(question)

    @staticmethod
    def _question(value: Any) -> str:
        """The question text of an ``interrupt()`` payload."""
        return str(value.get("question", value) if isinstance(value, dict) else value)

    async def _tts_worker(self, turn: _Turn) -> None:
        voices = self.runtime.voices
        loop = asyncio.get_running_loop()
        while (piece := await turn.queue.get()) is not None:
            if voices is None:
                continue
            self._status(Lane.SYNTHESIZING, LaneState.ACTIVE)
            audio, rate = await loop.run_in_executor(
                self.runtime.tts_executor, voices.synthesize, piece, turn.lang
            )
            if len(audio):
                if turn.first_audio is None:
                    turn.first_audio = time.monotonic()
                self.bus.publish(AudioEvent(float32_to_pcm16(audio), rate))
        self._status(Lane.SYNTHESIZING, LaneState.DONE)

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
            if not self.busy and not self.holding and self.runtime.voices is not None:
                prefix = "Emlékeztető: " if self.lang == "hu" else "Reminder: "
                loop = asyncio.get_running_loop()
                audio, rate = await loop.run_in_executor(
                    self.runtime.tts_executor,
                    self.runtime.voices.synthesize,
                    prefix + note.text,
                    self.lang,
                )
                if len(audio):
                    self.bus.publish(AudioEvent(float32_to_pcm16(audio), rate))
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
