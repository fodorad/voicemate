"""Kokoro-82M (ONNX) English voices.

``kokoro-onnx`` normally phonemizes through ``phonemizer`` + ``espeakng-loader``, whose
bundled espeak-ng ignores the data path on macOS and aborts the process. Piper ships a
working espeak-ng bridge producing the same IPA, so phonemes are generated with it and
passed to Kokoro directly (see docs/sdlc/spec.md §6).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

#: Kokoro's native output rate.
KOKORO_SAMPLE_RATE: int = 24_000


class KokoroSynthesizer:
    """A Kokoro voice.

    Args:
        model_path: ``kokoro-v1.0.onnx``.
        voices_path: ``voices-v1.0.bin``.
        voice: Voice style name, e.g. ``af_heart``.
        espeak_voice: espeak-ng language used for phonemes (``en-us`` or ``en-gb``).
        speed: Speaking rate multiplier.
    """

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: str = "af_heart",
        espeak_voice: str = "en-us",
        speed: float = 1.0,
    ) -> None:
        import onnxruntime
        from kokoro_onnx import Kokoro
        from piper.phonemize_espeak import EspeakPhonemizer

        session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._kokoro = Kokoro.from_session(session, str(voices_path))
        if voice not in self._kokoro.get_voices():
            raise ValueError(f"Unknown Kokoro voice {voice!r}")
        self._phonemizer = EspeakPhonemizer()
        self._vocab = self._kokoro.tokenizer.vocab
        self.voice = voice
        self.espeak_voice = espeak_voice
        self.speed = speed
        self.sample_rate: int = KOKORO_SAMPLE_RATE

    def phonemize(self, text: str) -> str:
        """IPA phonemes for ``text``, restricted to Kokoro's vocabulary."""
        sentences = self._phonemizer.phonemize(self.espeak_voice, text)
        joined = " ".join(unicodedata.normalize("NFC", "".join(s)) for s in sentences)
        return "".join(ch for ch in joined if ch in self._vocab).strip()

    def synthesize(self, text: str) -> np.ndarray:
        """Render ``text``; returns an empty array when there is nothing to say."""
        phonemes = self.phonemize(text)
        if not phonemes:
            return np.zeros(0, dtype=np.float32)
        audio, _ = self._kokoro.create(
            phonemes, voice=self.voice, speed=self.speed, is_phonemes=True
        )
        return np.asarray(audio, dtype=np.float32)
