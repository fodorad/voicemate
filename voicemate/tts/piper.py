"""Piper VITS voices (fast CPU inference; the Hungarian voices are the best local option)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PiperSynthesizer:
    """A Piper voice loaded from a local ``.onnx`` file (with ``.onnx.json`` next to it).

    Args:
        model_path: Path to the voice model.
    """

    def __init__(self, model_path: Path) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(model_path)
        self.sample_rate: int = self._voice.config.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Render ``text``; returns an empty array when there is nothing to say."""
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)
