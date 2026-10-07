"""Typed application configuration loaded from ``config/voicemate.toml``.

Every section of the TOML file maps onto one dataclass below. Missing keys fall back to
the dataclass defaults; unknown keys raise :class:`ValueError` so typos never pass silently.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Repository root (the directory that contains ``pyproject.toml``).
PROJECT_DIR: Path = Path(__file__).resolve().parents[1]

#: Default configuration file shipped with the repository.
DEFAULT_CONFIG_PATH: Path = PROJECT_DIR / "config" / "voicemate.toml"

#: Environment variable that points to an alternative configuration file.
CONFIG_ENV_VAR: str = "VOICEMATE_CONFIG"


def base_dir() -> Path:
    """Directory that relative paths are resolved against.

    The repository root in a checkout; otherwise (an installed package has no repository
    next to it) the current working directory.
    """
    return PROJECT_DIR if (PROJECT_DIR / "pyproject.toml").exists() else Path.cwd()


def _resolve(path: str) -> Path:
    """Expand ``~`` and resolve relative paths against :func:`base_dir`."""
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else base_dir() / expanded


@dataclass
class AssistantConfig:
    """Persona settings.

    Attributes:
        name: Persona name used in prompts and the UI.
        pronunciation: Per-language spelling the TTS voice should read instead of ``name``.
        default_language: Language used until the user's language is detected (``"en"`` or
            ``"hu"``); also the fallback voice.
    """

    name: str = "Ava"
    pronunciation: dict[str, str] = field(default_factory=dict)
    default_language: str = "en"


@dataclass
class LocationConfig:
    """Home location used for weather and local context.

    Attributes:
        name: Human-readable place name.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
    """

    name: str = "Gyöngyös, Heves megye, Hungary"
    latitude: float = 47.7826
    longitude: float = 19.9281


@dataclass
class LLMConfig:
    """Ollama chat model settings.

    Each profile is one Ollama model used for everything (chat, tools, memory decisions).

    Attributes:
        profile: Preferred profile name.
        profiles: Profile name → Ollama model tag.
        fallback_profile: Profile used when the preferred one is missing or does not fit.
        memory_guard: Check free memory at startup before choosing the model.
        reserve_gb: Memory kept free for the speech stack, the UI process and the OS (GB).
        temperature: Sampling temperature.
        num_ctx: Context window requested from Ollama.
        keep_alive: How long Ollama keeps the model loaded after the last request.
        max_tool_rounds: Maximum agent ⇄ tool iterations per turn.
        turn_timeout_s: Hard limit for one conversational turn.
    """

    profile: str = "fast"
    profiles: dict[str, str] = field(
        default_factory=lambda: {
            "fast": "gemma4:e4b",
            "gemma": "gemma4:26b-mlx",
            "qwen": "qwen3.8:27b-mlx",
        }
    )
    fallback_profile: str = "fast"
    memory_guard: bool = True
    reserve_gb: float = 5.0
    temperature: float = 0.6
    num_ctx: int = 32768
    keep_alive: str = "30m"
    max_tool_rounds: int = 6
    turn_timeout_s: float = 90.0

    @property
    def model(self) -> str:
        """Model tag of the preferred profile (before the memory check)."""
        return self.profiles[self.profile]


@dataclass
class ASRConfig:
    """Speech recognition settings.

    Attributes:
        model: Hugging Face repository of the Parakeet MLX checkpoint.
        partial_interval_s: Seconds of new speech between live partial transcripts.
        device: ``"cpu"`` or ``"gpu"`` for the MLX computation.
    """

    model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    partial_interval_s: float = 1.0
    device: str = "cpu"


@dataclass
class TTSVoiceConfig:
    """One TTS voice.

    Attributes:
        engine: ``"piper"`` or ``"kokoro"``.
        voice: Engine-specific voice identifier.
    """

    engine: str = "piper"
    voice: str = "hu_HU-anna-medium"


def _default_tts() -> dict[str, TTSVoiceConfig]:
    return {
        "hu": TTSVoiceConfig(engine="piper", voice="hu_HU-anna-medium"),
        "en": TTSVoiceConfig(engine="kokoro", voice="af_heart"),
    }


@dataclass
class VADConfig:
    """Voice activity detection and endpointing.

    Attributes:
        threshold: Speech probability above which a frame counts as speech.
        min_silence_ms: Trailing silence that ends an utterance.
        min_speech_ms: Minimum speech duration for an utterance to count.
        preroll_ms: Audio kept from before the detected speech start.
        barge_in_threshold: Stricter threshold used while the assistant speaks.
        barge_in_min_speech_ms: Stricter minimum speech used while the assistant speaks.
    """

    threshold: float = 0.5
    min_silence_ms: int = 700
    min_speech_ms: int = 250
    preroll_ms: int = 300
    barge_in_threshold: float = 0.8
    barge_in_min_speech_ms: int = 300


@dataclass
class MemoryConfig:
    """Long-term memory settings.

    Attributes:
        embed_model: Sentence-transformers model used for embeddings.
        search_cache_similarity: Query similarity for reusing a stored web search.
        fact_min_similarity: Minimum similarity for recalling a stored fact.
        recall_min_similarity: Minimum similarity for recalling episodes and research.
        fact_dedupe_similarity: Similarity above which a new fact replaces an old one.
        embed_device: ``"cpu"``, ``"mps"`` or ``"auto"`` for the embedding model.
    """

    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    search_cache_similarity: float = 0.86
    fact_min_similarity: float = 0.45
    recall_min_similarity: float = 0.55
    fact_dedupe_similarity: float = 0.9


@dataclass
class SearchConfig:
    """Web search backend.

    Attributes:
        backend: ``"ddgs"`` or ``"searxng"``.
        searxng_url: Base URL of a SearXNG instance.
        max_results: Number of results returned per search.
    """

    backend: str = "ddgs"
    searxng_url: str = "http://localhost:8888"
    max_results: int = 5


@dataclass
class FilesConfig:
    """File tool sandbox.

    Attributes:
        roots: Directories the file tools may read and write.
        papers_dir: Where downloaded papers are saved.
        max_read_bytes: Maximum text returned by one file read.
    """

    roots: list[str] = field(default_factory=lambda: ["~/Documents", "~/Downloads"])
    papers_dir: str = "~/Downloads/voicemate-papers"
    max_read_bytes: int = 200_000

    @property
    def root_paths(self) -> list[Path]:
        """Allowed roots as absolute paths."""
        return [_resolve(root) for root in self.roots]

    @property
    def papers_path(self) -> Path:
        """Papers directory as an absolute path."""
        return _resolve(self.papers_dir)


@dataclass
class UIConfig:
    """Web UI server.

    Attributes:
        host: Interface to bind; keep ``127.0.0.1`` for a private assistant.
        port: HTTP port.
        allowed_hosts: Accepted ``Host`` headers; anything else is rejected, which blocks DNS
            rebinding (a web page reaching the local assistant through a hostile domain).
        test_audio: Allow ``?fake_mic=<file>`` to play a WAV from ``data/fake_mic`` as if it
            were spoken (for automated end-to-end tests only).
    """

    host: str = "127.0.0.1"
    port: int = 8080
    allowed_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost"])
    test_audio: bool = False


@dataclass
class Config:
    """Root configuration.

    Attributes:
        data_dir: Directory for runtime state (memory, notes, checkpoints, logs, models).
        save_audio: Keep raw utterance audio under ``data/audio``.
    """

    data_dir: str = "data"
    save_audio: bool = False
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    location: LocationConfig = field(default_factory=LocationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: dict[str, TTSVoiceConfig] = field(default_factory=_default_tts)
    vad: VADConfig = field(default_factory=VADConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @property
    def data_path(self) -> Path:
        """Runtime data directory as an absolute path."""
        return _resolve(self.data_dir)


def _build(cls: Any, values: Mapping[str, Any], section: str) -> Any:
    """Instantiate dataclass ``cls`` from a mapping, recursing into nested sections."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) in [{section}]: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if name == "tts" and cls is Config:
            kwargs[name] = {
                lang: _build(TTSVoiceConfig, voice, f"tts.{lang}") for lang, voice in value.items()
            }
        elif is_dataclass(default) and isinstance(value, Mapping):
            kwargs[name] = _build(type(default), value, name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
    default_path: Path = DEFAULT_CONFIG_PATH,
) -> Config:
    """Load the configuration file.

    Args:
        path: Explicit TOML file. When omitted, ``$VOICEMATE_CONFIG`` or ``default_path`` is
            used; if the default file does not exist (installed package), the built-in
            defaults apply.
        env: Environment mapping (defaults to :data:`os.environ`).
        default_path: The repository's configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the file contains unknown keys.
        FileNotFoundError: If an explicitly named file does not exist.
    """
    env = os.environ if env is None else env
    if path is None and CONFIG_ENV_VAR in env:
        path = Path(env[CONFIG_ENV_VAR])
    if path is None:
        if not default_path.exists():
            logger.info("No %s; using built-in defaults", default_path.name)
            return Config()
        path = default_path
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return _build(Config, raw, "root")
