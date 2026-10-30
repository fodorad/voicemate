"""Ollama's Phonon voices (small, fast text-to-speech served by the local Ollama).

Phonon runs inside Ollama (Apple Silicon, Ollama 0.40.9 or newer) and is reached through its
OpenAI-compatible ``/v1/audio/speech`` endpoint. It speaks English, French, German, Spanish and
Portuguese (not Hungarian), so Hungarian keeps using Piper.
"""

from __future__ import annotations

import io
import logging

import httpx
import numpy as np
import soundfile

logger = logging.getLogger(__name__)

#: Model tag pulled with ``ollama pull``.
PHONON_MODEL: str = "phonon"

#: Phonon's output rate in Hz.
PHONON_SAMPLE_RATE: int = 24_000

#: First Ollama release that serves speech models.
MIN_OLLAMA_VERSION: tuple[int, int, int] = (0, 40, 9)

#: Where the local Ollama listens by default.
DEFAULT_HOST: str = "http://127.0.0.1:11434"


class PhononError(RuntimeError):
    """Phonon cannot be used; the message says what to fix."""


def _version_tuple(version: str) -> tuple[int, ...]:
    digits = version.split("-")[0].split("+")[0]
    return tuple(int(part) for part in digits.split(".") if part.isdigit())


class PhononSynthesizer:
    """A Phonon voice behind the local Ollama.

    Args:
        voice: Voice name from the model's voice list (see ``ollama show phonon``).
        host: Base URL of the Ollama server.
        timeout_s: Request timeout.

    Raises:
        PhononError: If Ollama is unreachable or older than :data:`MIN_OLLAMA_VERSION`, or the
            model has no such voice.
    """

    def __init__(
        self, voice: str = "default", host: str = DEFAULT_HOST, timeout_s: float = 30.0
    ) -> None:
        self.voice = voice
        self.sample_rate: int = PHONON_SAMPLE_RATE
        self._client = httpx.Client(base_url=host, timeout=timeout_s)
        self._check_version()
        self._check_voice()

    def _check_version(self) -> None:
        try:
            version = self._client.get("/api/version").json()["version"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise PhononError(f"Ollama is not reachable at {self._client.base_url}: {exc}") from exc
        if _version_tuple(version) < MIN_OLLAMA_VERSION:
            needed = ".".join(map(str, MIN_OLLAMA_VERSION))
            raise PhononError(f"Ollama {version} cannot speak: update to {needed} or newer")

    def _check_voice(self) -> None:
        try:
            response = self._client.post("/api/show", json={"model": PHONON_MODEL})
        except httpx.HTTPError as exc:
            raise PhononError(f"Ollama request failed: {exc}") from exc
        if response.status_code == 404:
            raise PhononError(f"Model {PHONON_MODEL!r} is missing: run `ollama pull phonon`")
        voices = response.json().get("voices", []) if response.status_code == 200 else []
        if voices and self.voice not in voices:
            available = ", ".join(voices)
            raise PhononError(f"Unknown phonon voice {self.voice!r}; available: {available}")

    def synthesize(self, text: str) -> np.ndarray:
        """Render ``text``; returns an empty array when there is nothing to say.

        Raises:
            PhononError: If the model is missing or Ollama rejects the request.
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        try:
            response = self._client.post(
                "/v1/audio/speech",
                json={
                    "model": PHONON_MODEL,
                    "input": text,
                    "voice": self.voice,
                    "response_format": "wav",
                },
            )
        except httpx.HTTPError as exc:
            raise PhononError(f"Ollama speech request failed: {exc}") from exc
        if response.status_code == 404:
            raise PhononError(f"Model {PHONON_MODEL!r} is missing: run `ollama pull phonon`")
        if response.status_code != 200:
            detail = response.text[:200]
            raise PhononError(f"Ollama speech failed ({response.status_code}): {detail}")
        audio, rate = soundfile.read(io.BytesIO(response.content), dtype="float32")
        if rate != self.sample_rate:
            raise PhononError(f"Unexpected sample rate {rate} Hz (expected {self.sample_rate})")
        return audio if audio.ndim == 1 else audio.mean(axis=1)
