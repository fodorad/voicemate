"""Benchmarks behind ``make bench``: ASR, TTS and LLM measurements (see docs/sdlc/spec.md §6).

Results are written to ``temp/bench/`` (gitignored). Test clips are generated with the macOS
``say`` command, so benchmarks run on macOS only.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from voicemate.config import base_dir

#: Output directory for benchmark artifacts.
BENCH_DIR: Path = base_dir() / "temp" / "bench"

#: Hungarian test sentences (assistant-style requests, includes the persona name).
HU_SENTENCES: tuple[str, ...] = (
    "Ava, mennyi az idő most?",
    "Milyen idő lesz holnap Gyöngyösön?",
    "Keress rá a legújabb transzformer architektúrákra.",
    "Ava, jegyezd fel, hogy holnap hívjam fel az orvost.",
    "Mit tudsz a diffúziós modellekről?",
    "Szeretnék gépi tanulási állást keresni Budapesten.",
    "Mennyi huszonhárom szorozva negyvenkettővel?",
    "Ava, milyen cikkeket olvastunk a múlt héten?",
    "Fáj a fejem, mit tehetnék otthon?",
    "Foglald össze a letöltött PDF fájl tartalmát.",
    "Köszönöm szépen, ennyi volt mára.",
    "Hány kalóriát égetek el egy óra futással?",
)

#: English test sentences.
EN_SENTENCES: tuple[str, ...] = (
    "Ava, what time is it right now?",
    "What will the weather be like tomorrow?",
    "Search for the latest papers on mixture of experts.",
    "Ava, add a note to call the dentist on Monday.",
    "Explain the difference between LoRA and full fine tuning.",
    "Find machine learning engineer jobs in Budapest.",
    "What is seventeen percent of three hundred and forty?",
    "Ava, what did we discuss about retrieval augmented generation?",
    "How much sleep does an adult really need?",
    "Summarize the paper I downloaded yesterday.",
    "Thanks, that is all for today.",
    "Which optimizer should I use for training a small transformer?",
)

#: macOS voices used to render the corpus.
SAY_VOICES: dict[str, str] = {"hu": "Tünde", "en": "Samantha"}


def normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace (for WER)."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def corpus() -> list[tuple[str, str]]:
    """``(lang, sentence)`` pairs."""
    return [("hu", s) for s in HU_SENTENCES] + [("en", s) for s in EN_SENTENCES]


def make_clips(folder: Path) -> list[tuple[str, str, Path]]:  # pragma: no cover - macOS only
    """Render the corpus to 16 kHz WAV files with ``say`` (cached)."""
    folder.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (lang, text) in enumerate(corpus()):
        path = folder / f"{lang}_{i:02d}.wav"
        if not path.exists():
            subprocess.run(
                ["say", "-v", SAY_VOICES[lang], "-o", str(path), "--data-format=LEI16@16000", text],
                check=True,
            )
        clips.append((lang, text, path))
    return clips
