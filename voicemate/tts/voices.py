"""Per-language voice selection."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from voicemate.config import Config
from voicemate.tts.base import Synthesizer
from voicemate.tts.normalize import speakable

logger = logging.getLogger(__name__)


class VoiceBank:
    """Routes text to the voice of its language and cleans it for speech first.

    Args:
        voices: Language code → synthesizer.
        pronunciation: Language code → whole-word replacements applied before synthesis.
        default_lang: Voice used for languages without their own voice.
    """

    def __init__(
        self,
        voices: Mapping[str, Synthesizer],
        pronunciation: Mapping[str, Mapping[str, str]] | None = None,
        default_lang: str = "hu",
    ) -> None:
        if default_lang not in voices:
            raise ValueError(f"Default language {default_lang!r} has no voice")
        self.voices = dict(voices)
        self.pronunciation = {k: dict(v) for k, v in (pronunciation or {}).items()}
        self.default_lang = default_lang

    def voice_for(self, lang: str) -> Synthesizer:
        """The synthesizer for ``lang`` (falls back to the default language)."""
        return self.voices.get(lang, self.voices[self.default_lang])

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        """Clean ``text`` and render it with the voice of ``lang``.

        Returns:
            ``(audio, sample_rate)``; audio is empty when nothing speakable remains.
        """
        voice = self.voice_for(lang)
        cleaned = speakable(text, self.pronunciation.get(lang))
        if not cleaned:
            return np.zeros(0, dtype=np.float32), voice.sample_rate
        return voice.synthesize(cleaned), voice.sample_rate

    @classmethod
    def from_config(cls, config: Config) -> VoiceBank:  # pragma: no cover - loads real models
        """Load the voices configured in ``[tts.*]``, downloading weights if missing."""
        from voicemate import models
        from voicemate.tts.kokoro import KokoroSynthesizer
        from voicemate.tts.piper import PiperSynthesizer

        voices: dict[str, Synthesizer] = {}
        for lang, voice in config.tts.items():
            logger.info("Loading %s voice %s (%s)", lang, voice.voice, voice.engine)
            if voice.engine == "piper":
                voices[lang] = PiperSynthesizer(models.ensure_piper_voice(config, voice.voice))
            elif voice.engine == "kokoro":
                model_path, voices_path = models.ensure_kokoro(config)
                voices[lang] = KokoroSynthesizer(model_path, voices_path, voice=voice.voice)
            else:
                raise ValueError(f"Unknown TTS engine {voice.engine!r} for [tts.{lang}]")
        name = config.assistant.name
        pronunciation = {
            lang: {name: spoken} for lang, spoken in config.assistant.pronunciation.items()
        }
        preferred = config.assistant.default_language
        default = preferred if preferred in voices else next(iter(voices))
        return cls(voices, pronunciation, default_lang=default)
