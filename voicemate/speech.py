"""Speech input and output behind one async interface.

:class:`Speech` pairs a speech recognizer with the text-to-speech voices and owns the worker
threads they run on. MLX models are not re-entrant, so each model gets exactly one worker
thread: callers simply ``await`` and never see an executor. Recognizers and voices are the
adapters (the real Parakeet/Piper/Kokoro models in production, small deterministic fakes in
tests).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from voicemate.asr.base import SpeechRecognizer
from voicemate.tts.voices import VoiceBank

logger = logging.getLogger(__name__)


class Speech:
    """Async transcription and synthesis, one worker thread per model.

    Args:
        recognizer: Turns 16 kHz mono float32 audio into text.
        voices: Per-language voices that clean and synthesize text.
    """

    def __init__(self, recognizer: SpeechRecognizer, voices: VoiceBank) -> None:
        self.recognizer = recognizer
        self.voices = voices
        self._asr = ThreadPoolExecutor(1, thread_name_prefix="asr")
        self._tts = ThreadPoolExecutor(1, thread_name_prefix="tts")
        self._closed = False

    async def transcribe(self, audio: np.ndarray) -> str:
        """The text spoken in ``audio`` (16 kHz mono float32).

        Raises:
            RuntimeError: If the speech module was closed.
        """
        self._check_open()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._asr, self.recognizer.transcribe_text, audio)

    async def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        """Render ``text`` with the voice of ``lang``.

        Returns:
            ``(audio, sample_rate)``; the audio is empty when nothing speakable remains.

        Raises:
            RuntimeError: If the speech module was closed.
        """
        self._check_open()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tts, self.voices.synthesize, text, lang)

    async def aclose(self) -> None:
        """Release the worker threads; later calls raise. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._asr.shutdown(wait=False, cancel_futures=True)
        self._tts.shutdown(wait=False, cancel_futures=True)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Speech is closed")
