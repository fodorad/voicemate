"""Choose the chat model so the whole pipeline fits into memory.

Each profile in ``[llm.profiles]`` is one Ollama model used for everything. On a 32 GB Mac a
model that does not fit next to the speech stack and the user's apps gets paged out, and a
reply then takes 15+ s instead of about 1 s (docs/sdlc/spec.md §10). At startup the preferred
profile is therefore checked against free memory and, if it does not fit, the fallback profile
is used and the user is told why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from voicemate.config import LLMConfig

logger = logging.getLogger(__name__)

#: Loaded size relative to the size on disk (context buffers, runtime). gemma4:e4b measured
#: 6.6 GB on disk and 11 GB loaded with a 32k context.
LOAD_FACTOR: float = 1.5

#: Fixed extra memory per loaded model (GB).
LOAD_OVERHEAD_GB: float = 1.0


@dataclass
class ModelChoice:
    """The model the runtime will use.

    Attributes:
        profile: Profile name the model belongs to.
        model: Ollama model tag.
        fits: Whether the estimate says it fits into free memory (False also when unknown).
        fell_back: True when the preferred profile was not used.
        reason: Human-readable explanation, shown in the log and the UI.
    """

    profile: str
    model: str
    fits: bool
    fell_back: bool
    reason: str


def normalize_name(model: str) -> str:
    """Ollama tags without an explicit version mean ``:latest``."""
    return model if ":" in model else f"{model}:latest"


def estimated_load_gb(disk_gb: float) -> float:
    """Estimated memory of a loaded model from its size on disk."""
    return disk_gb * LOAD_FACTOR + LOAD_OVERHEAD_GB


def choose_model(llm: LLMConfig, installed: dict[str, float], available_gb: float) -> ModelChoice:
    """Pick the preferred profile, or the fallback when it is missing or does not fit.

    Args:
        llm: LLM configuration (profiles, preferred and fallback profile, reserve).
        installed: Installed Ollama models (normalized tag → size on disk in GB).
        available_gb: Memory available for the model: free RAM plus memory of models Ollama
            would unload.

    Returns:
        The chosen model with an explanation.

    Raises:
        ValueError: If a configured profile name does not exist.
    """
    for name in (llm.profile, llm.fallback_profile):
        if name not in llm.profiles:
            raise ValueError(f"Unknown LLM profile {name!r}; known: {sorted(llm.profiles)}")
    order = [llm.profile] + ([llm.fallback_profile] if llm.fallback_profile != llm.profile else [])
    preferred = llm.profiles[llm.profile]
    if not llm.memory_guard:
        return ModelChoice(llm.profile, preferred, True, False, "memory guard disabled")

    problems: list[str] = []
    candidates: list[tuple[str, str, float]] = []
    for profile in order:
        model = llm.profiles[profile]
        size = installed.get(normalize_name(model))
        if size is None:
            problems.append(f"{model} is not installed (run `ollama pull {model}`)")
            continue
        need = estimated_load_gb(size) + llm.reserve_gb
        candidates.append((profile, model, need))
        if need <= available_gb:
            fell_back = profile != llm.profile
            reason = "; ".join(
                [*problems, f"{model} needs about {need:.1f} GB, {available_gb:.1f} GB available"]
            )
            return ModelChoice(profile, model, True, fell_back, reason)
        problems.append(f"{model} needs about {need:.1f} GB but only {available_gb:.1f} GB is free")

    if not candidates:
        return ModelChoice(llm.profile, preferred, False, False, "; ".join(problems))
    profile, model, _ = min(candidates, key=lambda item: item[2])
    reason = "; ".join(problems) + ". Expect slow replies; close other apps to free memory."
    return ModelChoice(profile, model, False, profile != llm.profile, reason)


def probe_and_choose(llm: LLMConfig) -> ModelChoice:  # pragma: no cover - needs Ollama + RAM
    """Query Ollama and the OS, then :func:`choose_model`."""
    import ollama
    import psutil

    gb = 1024**3
    installed = {normalize_name(m.model or ""): (m.size or 0) / gb for m in ollama.list().models}
    # Models Ollama currently holds are unloaded on demand, so their memory counts as available.
    loaded = sum((m.size or 0) for m in ollama.ps().models) / gb
    available = psutil.virtual_memory().available / gb + loaded
    choice = choose_model(llm, installed, available)
    log = logger.info if choice.fits and not choice.fell_back else logger.warning
    log("LLM profile %s → %s (%s)", choice.profile, choice.model, choice.reason)
    return choice
