"""Voice activity detection and utterance segmentation (endpointing).

:class:`Segmenter` turns a continuous 16 kHz microphone stream into utterances. It is
independent of the speech-probability model: :class:`SileroVAD` is used at runtime and the
deterministic :class:`EnergyVAD` is handy for tests and as a dependency-free fallback.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

import numpy as np

from voicemate.audio.codec import SAMPLE_RATE
from voicemate.config import VADConfig

#: Hysteresis: a frame keeps an utterance alive while its probability is above
#: ``threshold - NEGATIVE_MARGIN`` (the recipe recommended by Silero).
NEGATIVE_MARGIN: float = 0.15

#: Trailing silence kept at the end of an utterance (ms); the rest is trimmed.
TAIL_KEEP_MS: int = 100


class SpeechProbability(Protocol):
    """A frame-level speech probability model."""

    #: Number of 16 kHz samples per frame the model expects.
    frame_size: int

    def __call__(self, frame: np.ndarray) -> float:
        """Return the speech probability of one frame in ``[0, 1]``."""
        ...

    def reset(self) -> None:
        """Clear recurrent state between independent streams."""
        ...


class EnergyVAD:
    """RMS-energy speech probability. Deterministic; no model needed.

    Args:
        reference_rms: RMS level mapped to probability 1.0.
        frame_size: Samples per frame.
    """

    def __init__(self, reference_rms: float = 0.05, frame_size: int = 512) -> None:
        self.reference_rms = reference_rms
        self.frame_size = frame_size

    def __call__(self, frame: np.ndarray) -> float:
        """Return ``min(1, rms / reference_rms)``."""
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        return min(1.0, rms / self.reference_rms)

    def reset(self) -> None:
        """Stateless; nothing to reset."""


def silero_model_path() -> Path:
    """Locate the ONNX model shipped in the ``silero-vad`` wheel without importing it.

    Importing the package would pull in torch, which costs seconds and is not needed here.
    """
    spec = find_spec("silero_vad")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("silero-vad is not installed")
    return Path(spec.origin).parent / "data" / "silero_vad.onnx"


class SileroVAD:
    """Silero VAD v6 run directly with onnxruntime (16 kHz, 512-sample frames).

    The ONNX file is the one bundled with the ``silero-vad`` package.
    """

    frame_size: int = 512
    _CONTEXT = 64

    def __init__(self) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        model_path = silero_model_path()
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear the recurrent state and the audio context."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Return the speech probability of one 512-sample frame.

        Raises:
            ValueError: If the frame does not have exactly 512 samples.
        """
        if frame.shape != (self.frame_size,):
            raise ValueError(f"Silero expects {self.frame_size} samples, got {frame.shape}")
        x = np.concatenate([self._context, frame[None, :].astype(np.float32)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -self._CONTEXT :]
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class SpeechStart:
    """The user started speaking."""


@dataclass
class SpeechEnd:
    """The user finished an utterance.

    Attributes:
        audio: The utterance (16 kHz float32) including pre-roll and a short tail.
    """

    audio: np.ndarray


#: Events produced by :meth:`Segmenter.feed`.
SegmentEvent = SpeechStart | SpeechEnd


class Segmenter:
    """Endpointing state machine: idle → candidate → speech → idle.

    Args:
        vad: Frame-level speech probability model.
        config: Thresholds and durations.
        max_utterance_s: Utterances longer than this are force-ended.
    """

    def __init__(self, vad: SpeechProbability, config: VADConfig, max_utterance_s: float = 30.0):
        self.vad = vad
        self.config = config
        self.max_samples = int(max_utterance_s * SAMPLE_RATE)
        #: While True (the assistant is speaking) stricter barge-in thresholds apply.
        self.strict = False
        self._frame_ms = 1000 * vad.frame_size / SAMPLE_RATE
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, math.ceil(config.preroll_ms / self._frame_ms))
        )
        self._pending = np.zeros(0, dtype=np.float32)
        self.reset()

    # ── public state ──────────────────────────────────────────────────────────

    @property
    def in_speech(self) -> bool:
        """True between :class:`SpeechStart` and :class:`SpeechEnd`."""
        return self._state == "speech"

    @property
    def current_audio(self) -> np.ndarray:
        """Audio of the utterance in progress (empty when idle)."""
        if self._state != "speech" or not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    @property
    def speech_seconds(self) -> float:
        """Duration of the utterance in progress."""
        return sum(len(f) for f in self._frames) / SAMPLE_RATE if self.in_speech else 0.0

    def reset(self) -> None:
        """Drop any utterance in progress and reset the model state."""
        self._state = "idle"
        self._frames: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._preroll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self.vad.reset()

    def flush(self) -> SpeechEnd | None:
        """End the input now (the microphone was turned off).

        Returns:
            The utterance in progress, or None when there was no speech (a blip that never
            reached ``min_speech_ms`` is dropped). The segmenter is reset either way.
        """
        end = self._finish() if self.in_speech and self._frames else None
        self.reset()
        return end

    # ── streaming ─────────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray) -> list[SegmentEvent]:
        """Consume a chunk of 16 kHz float32 audio of any length.

        Returns:
            Events triggered by this chunk, in order.
        """
        buffer = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32)])
        size = self.vad.frame_size
        n_frames = len(buffer) // size
        self._pending = buffer[n_frames * size :]
        events: list[SegmentEvent] = []
        for i in range(n_frames):
            event = self._step(buffer[i * size : (i + 1) * size])
            if event is not None:
                events.append(event)
        return events

    def _thresholds(self) -> tuple[float, float]:
        if self.strict:
            return self.config.barge_in_threshold, self.config.barge_in_min_speech_ms
        return self.config.threshold, self.config.min_speech_ms

    def _step(self, frame: np.ndarray) -> SegmentEvent | None:
        threshold, min_speech_ms = self._thresholds()
        prob = self.vad(frame)
        is_speech = prob >= threshold
        still_speech = prob >= threshold - NEGATIVE_MARGIN

        if self._state == "idle":
            if is_speech:
                self._state = "candidate"
                self._frames = [*self._preroll, frame]
                self._speech_ms, self._silence_ms = self._frame_ms, 0.0
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        if self._state == "candidate":
            if is_speech:
                self._speech_ms += self._frame_ms
                self._silence_ms = 0.0
            else:
                self._silence_ms += self._frame_ms
            if self._speech_ms >= min_speech_ms:
                self._state = "speech"
                self._silence_ms = 0.0
                return SpeechStart()
            if self._silence_ms >= min_speech_ms:
                self._abandon()
            return None

        # state == "speech"
        self._silence_ms = 0.0 if still_speech else self._silence_ms + self._frame_ms
        total = sum(len(f) for f in self._frames)
        if self._silence_ms >= self.config.min_silence_ms or total >= self.max_samples:
            return self._finish()
        return None

    def _abandon(self) -> None:
        self._state = "idle"
        self._preroll.clear()
        self._preroll.extend(self._frames[-self._preroll.maxlen :] if self._preroll.maxlen else [])
        self._frames = []

    def _finish(self) -> SpeechEnd:
        audio = np.concatenate(self._frames)
        trim = int(max(0.0, self._silence_ms - TAIL_KEEP_MS) * SAMPLE_RATE / 1000)
        if trim:
            audio = audio[:-trim]
        self._state = "idle"
        self._frames = []
        self._preroll.clear()
        self._speech_ms = self._silence_ms = 0.0
        return SpeechEnd(audio)
