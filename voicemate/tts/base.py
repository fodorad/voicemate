"""Text-to-speech interface shared by all engines."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Synthesizer(Protocol):
    """A TTS voice that renders one chunk of text at a time."""

    #: Output sample rate in Hz.
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Render ``text`` to mono float32 audio at :attr:`sample_rate`."""
        ...
