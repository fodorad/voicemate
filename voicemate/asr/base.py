"""Speech recognition interface shared by all ASR backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Transcript:
    """Result of transcribing one utterance.

    Attributes:
        text: Recognized text (stripped).
        lang: Language code (``"hu"`` or ``"en"``).
        audio_seconds: Duration of the input audio.
        latency_seconds: Wall-clock time the recognition took.
    """

    text: str
    lang: str
    audio_seconds: float
    latency_seconds: float


class SpeechRecognizer(Protocol):
    """Anything that turns 16 kHz mono float32 audio into text."""

    def transcribe_text(self, audio: np.ndarray) -> str:
        """Return the raw transcript of ``audio``."""
        ...
