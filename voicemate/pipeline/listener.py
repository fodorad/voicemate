"""Listening: microphone frames in, partial transcripts and final utterances out.

:class:`Listener` wraps the voice-activity :class:`~voicemate.audio.vad.Segmenter`. Speech
start and end are returned from :meth:`Listener.feed` immediately; the slower work
(transcribing a finished utterance, live partial transcripts while the user is still speaking)
runs in the background and is reported through :class:`ListenerCallbacks`. The module keeps
track of what is still in flight, so the session can ask "is everything transcribed?"
(:meth:`Listener.settled`) when a hold is released, and can end a sentence on demand
(:meth:`Listener.flush`) when the microphone is switched off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from voicemate.audio.codec import SAMPLE_RATE, pcm16_to_float32
from voicemate.audio.vad import Segmenter, SegmentEvent, SpeechEnd, SpeechProbability
from voicemate.config import VADConfig
from voicemate.events import LaneState

logger = logging.getLogger(__name__)

#: Live partial transcripts only cover the most recent speech, so a long utterance never
#: makes the final transcript wait behind a slow partial pass on the shared ASR thread.
PARTIAL_WINDOW_S: float = 8.0


@dataclass(frozen=True)
class Utterance:
    """One transcribed utterance.

    Attributes:
        text: What was recognized (not stripped).
        audio_seconds: Duration of the audio.
        latency_s: Seconds the recognition took.
        ended_at: Monotonic time the user finished speaking.
    """

    text: str
    audio_seconds: float
    latency_s: float
    ended_at: float


@dataclass(frozen=True)
class ListenerCallbacks:
    """Where the listener reports background work.

    Attributes:
        on_partial: A live transcript of speech still in progress (prefixed ``"… "`` when it
            covers only the end of a long utterance).
        on_transcribing: Transcription of a finished utterance is active, done or failed
            (idle).
        on_utterance: A finished utterance with words in it; the listener counts it as in
            flight until this returns.
        on_error: Recognition failed.
    """

    on_partial: Callable[[str], None]
    on_transcribing: Callable[[LaneState], None]
    on_utterance: Callable[[Utterance], Awaitable[None]]
    on_error: Callable[[Exception], None]


class Listener:
    """Frames of microphone audio to utterances.

    Args:
        vad: Frame-level speech probability model.
        vad_config: Endpointing thresholds.
        transcribe: Turns 16 kHz mono float32 audio into text.
        spawn: Starts a tracked background task (the session keeps strong references).
        callbacks: Receives partials, transcription state, utterances and errors.
        partial_interval_s: Seconds of new speech between live partial transcripts.
        partial_window_s: How much of the most recent speech a partial transcript covers.
        audio_dir: Where to save utterance audio (not saved when None).
    """

    def __init__(
        self,
        vad: SpeechProbability,
        vad_config: VADConfig,
        transcribe: Callable[[np.ndarray], Awaitable[str]],
        spawn: Callable[[Coroutine[Any, Any, Any]], asyncio.Task],
        callbacks: ListenerCallbacks,
        partial_interval_s: float = 1.0,
        partial_window_s: float = PARTIAL_WINDOW_S,
        audio_dir: Path | None = None,
    ) -> None:
        self.segmenter = Segmenter(vad, vad_config)
        self._transcribe = transcribe
        self._spawn = spawn
        self._callbacks = callbacks
        self._partial_interval_s = partial_interval_s
        self._partial_window_s = partial_window_s
        self._audio_dir = audio_dir
        self._partial: asyncio.Task | None = None
        self._last_partial_at = 0.0
        self._in_flight: set[asyncio.Task] = set()

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def in_speech(self) -> bool:
        """True while the user is speaking."""
        return self.segmenter.in_speech

    @property
    def strict(self) -> bool:
        """Stricter barge-in thresholds apply while the assistant speaks."""
        return self.segmenter.strict

    @strict.setter
    def strict(self, value: bool) -> None:
        self.segmenter.strict = value

    # ── input ─────────────────────────────────────────────────────────────────

    def feed(self, pcm: bytes) -> list[SegmentEvent]:
        """Consume one chunk of 16 kHz mono PCM16 audio.

        Returns:
            The speech start and end events it triggered, in order. A finished utterance is
            transcribed in the background.
        """
        events = self.segmenter.feed(pcm16_to_float32(pcm))
        for event in events:
            if isinstance(event, SpeechEnd):
                self._utterance_ended(event)
        self._maybe_partial()
        return events

    def flush(self) -> SpeechEnd | None:
        """End the sentence in progress now (the microphone was switched off).

        Returns:
            The end event, or None when the user was not speaking.
        """
        end = self.segmenter.flush()
        if end is not None:
            self._utterance_ended(end)
        return end

    async def settled(self) -> None:
        """Wait until no utterance is being transcribed or handed on."""
        while self._in_flight:
            await asyncio.wait(set(self._in_flight))

    def stop(self) -> None:
        """Cancel the background work that is still pending."""
        for task in [*self._in_flight, *([self._partial] if self._partial else [])]:
            task.cancel()

    # ── background work ───────────────────────────────────────────────────────

    def _utterance_ended(self, event: SpeechEnd) -> None:
        self._last_partial_at = 0.0
        task = self._spawn(self._handle(event.audio))
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)

    def _maybe_partial(self) -> None:
        """Schedule a live partial transcript every ``partial_interval_s`` of speech."""
        if not self.segmenter.in_speech or (self._partial and not self._partial.done()):
            return
        seconds = self.segmenter.speech_seconds
        if seconds - self._last_partial_at < self._partial_interval_s:
            return
        self._last_partial_at = seconds
        audio = self.segmenter.current_audio
        window = int(self._partial_window_s * SAMPLE_RATE)
        self._partial = self._spawn(self._publish_partial(audio[-window:], len(audio) > window))

    async def _publish_partial(self, audio: np.ndarray, truncated: bool) -> None:
        text = await self._transcribe(audio)
        if text and self.segmenter.in_speech:
            self._callbacks.on_partial(f"… {text}" if truncated else text)

    async def _handle(self, audio: np.ndarray) -> None:
        ended_at = time.monotonic()
        self._callbacks.on_transcribing(LaneState.ACTIVE)
        try:
            text = await self._transcribe(audio)
        except Exception as exc:
            logger.exception("Speech recognition failed")
            self._callbacks.on_transcribing(LaneState.IDLE)
            self._callbacks.on_error(exc)
            return
        latency = time.monotonic() - ended_at
        self._callbacks.on_transcribing(LaneState.DONE)
        if self._audio_dir is not None:
            await asyncio.to_thread(self._save, audio)
        if not any(ch.isalnum() for ch in text):
            return
        await self._callbacks.on_utterance(
            Utterance(text, len(audio) / SAMPLE_RATE, latency, ended_at)
        )

    def _save(self, audio: np.ndarray) -> None:
        import soundfile as sf

        assert self._audio_dir is not None
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        sf.write(self._audio_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}.wav", audio, SAMPLE_RATE)
