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
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any

import numpy as np

from voicemate.agent.hitl import Confirmation
from voicemate.asr.base import Transcript
from voicemate.audio.codec import float32_to_pcm16
from voicemate.audio.vad import SpeechEnd, SpeechStart
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
from voicemate.memory.notes import Note
from voicemate.pipeline.conversation import (
    ChatLine,
    Conversation,
)
from voicemate.pipeline.listener import Listener, ListenerCallbacks, Utterance
from voicemate.pipeline.reminders import Reminders, spoken_text
from voicemate.pipeline.reply import Reply
from voicemate.pipeline.turn_taking import Begin, Collect, Decision, Interrupt, TurnTaking
from voicemate.runtime import Runtime

logger = logging.getLogger(__name__)


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
        self.listener = Listener(
            runtime.vad_factory(),
            self.config.vad,
            self._transcribe,
            self._spawn,
            ListenerCallbacks(
                on_partial=self._on_partial,
                on_transcribing=self._on_transcribing,
                on_utterance=self._on_utterance,
                on_error=self._on_recognition_error,
            ),
            partial_interval_s=self.config.asr.partial_interval_s,
            audio_dir=self.config.data_path / "audio" if self.config.save_audio else None,
        )
        self.lang = self.config.assistant.default_language
        #: The rules for what user input means (hold, merge, confirmation).
        self.policy = TurnTaking()
        self._speaking = False
        self._turn: asyncio.Task | None = None
        self._reminders: asyncio.Task | None = None
        self._reminders_due = (
            Reminders(runtime.notes, self._announce_reminder) if runtime.notes else None
        )
        self._interrupted_reply = ""  # partial reply of the turn cut off by the last barge-in
        self._tasks: set[asyncio.Task] = set()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background duties (reminder watcher)."""
        if self._reminders_due is not None and self._reminders is None:
            self._reminders = self._spawn(self._reminders_due.watch())

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
    def holding(self) -> bool:
        """True while the user holds: pauses do not end their turn (see :meth:`hold`)."""
        return self.policy.holding

    @property
    def confirmation(self) -> Confirmation | None:
        """The question the agent is waiting on (human-in-the-loop), if any."""
        question = self.policy.question
        return Confirmation(question) if question is not None else None

    @property
    def _turn_running(self) -> bool:
        return self._turn is not None and not self._turn.done()

    @property
    def busy(self) -> bool:
        """True while a turn is being processed or its audio is playing."""
        return self._speaking or self._turn_running

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
        for event in self.listener.feed(pcm):
            if isinstance(event, SpeechStart):
                self._status(Lane.LISTENING, LaneState.ACTIVE)
                if isinstance(self.policy.on_speech_start(self.busy), Interrupt):
                    await self.interrupt()
            elif isinstance(event, SpeechEnd):
                self._status(Lane.LISTENING, LaneState.DONE)

    async def submit_text(self, text: str) -> None:
        """Handle typed input like a spoken utterance (no ASR, never merged)."""
        text = text.strip()
        if text:
            now = time.monotonic()
            await self._input(lambda running: self.policy.on_typed(text, now, running), now)

    async def answer_confirmation(self, answer: str) -> None:
        """Answer the pending human-in-the-loop question (e.g. from a UI button)."""
        now = time.monotonic()
        await self._input(lambda running: self.policy.on_answer(answer, now, running), now)

    async def interrupt(self) -> None:
        """Barge-in: stop the current turn and tell the browser to stop playback."""
        self.bus.publish(InterruptEvent())
        self._speaking = False
        self.listener.strict = False
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
        self.listener.strict = True
        self._status(Lane.SPEAKING, LaneState.ACTIVE)

    def playback_ended(self) -> None:
        """The browser finished playing all queued audio."""
        self._speaking = False
        self.listener.strict = False
        self._status(Lane.SPEAKING, LaneState.DONE)

    def microphone_stopped(self) -> None:
        """The user turned the microphone off: answer what was heard so far.

        A sentence still in progress is ended here, and a hold is released, because turning
        the microphone off means "I'm done talking".
        """
        end = self.listener.flush()
        self._status(Lane.LISTENING, LaneState.IDLE if end is None else LaneState.DONE)
        if self.holding:
            self._spawn(self.release())

    async def hold(self) -> None:
        """Keep listening until :meth:`release`; pauses no longer end the user's turn.

        A running answer is stopped first: the user wants the floor.
        """
        decision = self.policy.hold(self.busy)
        if isinstance(decision, Interrupt):
            await self.interrupt()
        self.bus.publish(HoldEvent(True))

    async def release(self) -> None:
        """The user is finished: send everything said (or typed) during the hold."""
        if not self.holding:
            return
        if self.listener.flush() is not None:
            self._status(Lane.LISTENING, LaneState.DONE)
        await self.listener.settled()  # transcripts still on their way join the question
        await self._input(self.policy.release, time.monotonic())
        self.bus.publish(HoldEvent(False))

    async def restore(self) -> list[ChatLine]:
        """Pick up this thread where it stopped (a reopened page).

        Reopens a question the agent is still waiting on and returns what was said so far.
        """
        if question := await self.conversation.pending_question():
            self.policy.question_asked(question)
            self.bus.publish(ConfirmEvent(question))
        return await self.conversation.lines()

    # ── ASR ──────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        speech = self.runtime.speech
        return "" if speech is None else await speech.transcribe(audio)

    def _on_partial(self, text: str) -> None:
        self.bus.publish(TranscriptEvent(text, final=False))

    def _on_transcribing(self, state: LaneState) -> None:
        self._status(Lane.TRANSCRIBING, state)

    def _on_recognition_error(self, error: Exception) -> None:
        self.bus.publish(ErrorEvent(f"Speech recognition failed: {error}"))

    async def _on_utterance(self, utterance: Utterance) -> None:
        transcript = Transcript(utterance.text, "", utterance.audio_seconds, utterance.latency_s)
        await self._input(
            lambda running: self.policy.on_utterance(
                utterance.text.strip(), utterance.ended_at, running
            ),
            utterance.ended_at,
            transcript,
        )

    # ── turns ────────────────────────────────────────────────────────────────

    async def _input(
        self,
        decide: Callable[[bool], Decision],
        ended_at: float,
        transcript: Transcript | None = None,
    ) -> None:
        """The single entry point for user input: at most one turn runs at a time.

        Args:
            decide: Asks the turn-taking policy what the input means, given whether a turn
                is running.
            ended_at: Monotonic time the user finished (latency reference).
            transcript: ASR details when the input was spoken.
        """
        decision = decide(self._turn_running)
        if isinstance(decision, Interrupt):  # the running turn stops, then the input counts
            await self.interrupt()
            decision = decide(False)
        match decision:
            case Collect(text, parts):
                self.lang = self.runtime.langid.detect(text, previous=self.lang)
                self.bus.publish(
                    TranscriptEvent(text, final=True, lang=self.lang, merged=parts > 1)
                )
            case Begin():
                await self._begin_turn(decision, ended_at, transcript)

    async def _begin_turn(
        self, begin: Begin, ended_at: float, transcript: Transcript | None
    ) -> None:
        if begin.resume is not None:
            self.bus.publish(ConfirmEvent("", open=False))
        if begin.discard is not None:
            await self.conversation.discard(begin.discard)
        text = begin.text
        lang = self.runtime.langid.detect(text, previous=self.lang)
        self.lang = lang
        self.bus.publish(
            TranscriptEvent(text, final=True, lang=lang, merged=begin.merged or begin.shown)
        )
        self._turn = self._spawn(self._run_turn(text, lang, ended_at, transcript, begin.resume))

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
        reply = Reply(lang, ended_at)
        message_id = uuid.uuid4().hex
        emit = self._tool_emitter(reply)
        chunks = (
            self.conversation.resume(resume, emit)
            if resume is not None
            else self.conversation.say(text, lang, message_id, emit)
        )
        worker = asyncio.create_task(self._tts_worker(reply))
        self._status(Lane.THINKING, LaneState.ACTIVE)
        try:
            async with asyncio.timeout(self.config.llm.turn_timeout_s):
                await self._stream(reply, chunks)
                await self._ask_if_paused(reply)
        except asyncio.CancelledError:
            worker.cancel()
            if reply.text:
                self._interrupted_reply = reply.text
                self.bus.publish(ReplyEvent(reply.text, interrupted=True))
            self.policy.turn_cancelled(
                message_id,
                text,
                time.monotonic(),
                spoke=bool(reply.text),
                resumed=resume is not None,
            )
            raise
        except Exception as exc:  # timeout, Ollama down, ...
            logger.exception("Turn failed")
            self.bus.publish(ErrorEvent(f"Turn failed: {reply.fail(exc)}"))
        self._status(Lane.THINKING, LaneState.IDLE)
        reply.finish()
        self._status(Lane.GENERATING, LaneState.DONE)
        await worker
        self.bus.publish(ReplyEvent(reply.text))
        self._publish_metrics(reply, transcript)
        if reply.text and self.runtime.memory is not None and not reply.failed:
            memory = self.runtime.memory
            self._spawn(
                asyncio.to_thread(memory.episodes.add, text, reply.text, lang, self.thread_id)
            )

    def _tool_emitter(self, reply: Reply):
        """The ``emit`` callback for one turn: publishes events and tells the reply."""

        def on_event(event: Event) -> None:
            self.emit(event)
            if isinstance(event, ToolEvent) and event.phase == "start":
                if reply.on_tool_start(event.name):
                    self._status(Lane.THINKING, LaneState.DONE)

        return on_event

    async def _stream(self, reply: Reply, chunks: AsyncIterator[str]) -> None:
        """Feed the agent's streamed answer into the reply and the UI."""
        async for content in chunks:
            if reply.mark_first_token():
                self._status(Lane.THINKING, LaneState.DONE)
            if reply.add_text(content):
                self._status(Lane.GENERATING, LaneState.ACTIVE)
            self.bus.publish(TokenEvent(content))

    async def _ask_if_paused(self, reply: Reply) -> None:
        """If a tool paused the graph for permission, show and speak the question."""
        question = await self.conversation.pending_question()
        if question is None:
            return
        self.policy.question_asked(question)
        self.bus.publish(ConfirmEvent(question))
        reply.say(question)

    async def _tts_worker(self, reply: Reply) -> None:
        while (piece := await reply.queue.get()) is not None:
            if self.runtime.speech is None:
                continue
            self._status(Lane.SYNTHESIZING, LaneState.ACTIVE)
            if await self._speak(piece, reply.lang):
                reply.audio_started()
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

    def _publish_metrics(self, reply: Reply, transcript: Transcript | None) -> None:
        metrics = reply.metrics(transcript)
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
        """Announce due reminders once; returns how many were announced."""
        return await self._reminders_due.check() if self._reminders_due else 0

    async def _announce_reminder(self, note: Note) -> None:
        """A reminder is always shown in the UI; it is also spoken when the assistant is idle."""
        self.bus.publish(ReminderEvent(note.id, note.text))
        if not self.busy and not self.holding:
            await self._speak(spoken_text(note, self.lang), self.lang)


def turns_log_path(data_path: Path) -> Path:
    """Location of the per-turn metrics log."""
    return data_path / "logs" / "turns.jsonl"
