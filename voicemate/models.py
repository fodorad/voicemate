"""Locate and download model assets.

Large weights never live in git. Piper and Kokoro files are stored under
``<data_dir>/models``; Hugging Face checkpoints (Parakeet, bge-m3) use the standard HF cache;
the LLM is pulled into Ollama's own store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from voicemate.config import Config

logger = logging.getLogger(__name__)

#: Hugging Face repository with all Piper voices.
PIPER_REPO: str = "rhasspy/piper-voices"

#: Kokoro ONNX release assets (Apache-2.0 model weights, packaged by kokoro-onnx).
KOKORO_BASE_URL: str = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)

#: Kokoro file names.
KOKORO_MODEL_FILE: str = "kokoro-v1.0.onnx"

#: Kokoro voice-style file name.
KOKORO_VOICES_FILE: str = "voices-v1.0.bin"


@dataclass(frozen=True)
class PiperVoiceId:
    """Parsed Piper voice name such as ``hu_HU-anna-medium``.

    Attributes:
        locale: Locale, e.g. ``hu_HU``.
        name: Speaker name, e.g. ``anna``.
        quality: ``x_low``, ``low``, ``medium`` or ``high``.
    """

    locale: str
    name: str
    quality: str

    @classmethod
    def parse(cls, voice: str) -> PiperVoiceId:
        """Parse ``<locale>-<name>-<quality>``.

        Raises:
            ValueError: If the name does not have three dash-separated parts.
        """
        parts = voice.split("-")
        if len(parts) != 3 or "_" not in parts[0]:
            raise ValueError(f"Piper voice must look like 'hu_HU-anna-medium', got {voice!r}")
        return cls(*parts)

    @property
    def repo_path(self) -> str:
        """Path of the ``.onnx`` file inside :data:`PIPER_REPO`."""
        lang = self.locale.split("_")[0]
        return f"{lang}/{self.locale}/{self.name}/{self.quality}/{self}.onnx"

    def __str__(self) -> str:
        """The canonical ``<locale>-<name>-<quality>`` name."""
        return f"{self.locale}-{self.name}-{self.quality}"


def models_dir(config: Config) -> Path:
    """Directory for downloaded TTS weights."""
    return config.data_path / "models"


def piper_voice_path(config: Config, voice: str) -> Path:
    """Local path of a Piper voice model (its ``.json`` config sits next to it)."""
    return models_dir(config) / "piper" / f"{PiperVoiceId.parse(voice)}.onnx"


def kokoro_paths(config: Config) -> tuple[Path, Path]:
    """Local paths of the Kokoro model and voice-style files."""
    base = models_dir(config) / "kokoro"
    return base / KOKORO_MODEL_FILE, base / KOKORO_VOICES_FILE


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_bytes(1 << 20):
                handle.write(block)
    partial.rename(target)


def ensure_piper_voice(config: Config, voice: str) -> Path:
    """Download a Piper voice (model + JSON config) if missing and return its path."""
    path = piper_voice_path(config, voice)
    voice_id = PiperVoiceId.parse(voice)
    base = f"https://huggingface.co/{PIPER_REPO}/resolve/main/{voice_id.repo_path}"
    for suffix, url in ((".onnx", base), (".onnx.json", base + ".json")):
        target = path.with_suffix(suffix)
        if not target.exists():
            _download(url, target)
    return path


def ensure_kokoro(config: Config) -> tuple[Path, Path]:
    """Download the Kokoro model and voices if missing and return their paths."""
    paths = kokoro_paths(config)
    for path in paths:
        if not path.exists():
            _download(f"{KOKORO_BASE_URL}/{path.name}", path)
    return paths


def ensure_all(config: Config) -> None:  # pragma: no cover - network + large downloads
    """Fetch every model the configured pipeline needs (``make models``)."""
    import ollama
    from huggingface_hub import snapshot_download

    for voice in config.tts.values():
        if voice.engine == "piper":
            ensure_piper_voice(config, voice.voice)
        elif voice.engine == "kokoro":
            ensure_kokoro(config)
    logger.info("Fetching ASR checkpoint %s", config.asr.model)
    snapshot_download(config.asr.model)
    logger.info("Fetching embedding model %s", config.memory.embed_model)
    snapshot_download(config.memory.embed_model)
    for model in sorted({config.llm.model, config.llm.profiles[config.llm.fallback_profile]}):
        logger.info("Pulling Ollama model %s (this can take a while)", model)
        ollama.pull(model)
