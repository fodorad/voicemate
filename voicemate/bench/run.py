"""Benchmark runners (ASR, TTS round trip, LLM tool calling). macOS + downloaded models only."""

from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicemate.bench import BENCH_DIR, corpus, make_clips, normalize
from voicemate.config import Config

logger = logging.getLogger(__name__)


def _wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return float(jiwer.wer(refs, hyps))


def _save(name: str, result: dict) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("%s → %s", json.dumps(result, ensure_ascii=False), path)
    return path


def bench_asr(config: Config) -> dict:
    """WER and real-time factor of the configured ASR per language."""
    from voicemate.asr.parakeet import ParakeetRecognizer

    asr = ParakeetRecognizer(config.asr.model)
    per_lang: dict[str, dict] = {}
    for lang, text, path in make_clips(BENCH_DIR / "clips"):
        audio, _ = sf.read(path, dtype="float32")
        start = time.perf_counter()
        hypothesis = asr.transcribe_text(audio)
        elapsed = time.perf_counter() - start
        entry = per_lang.setdefault(lang, {"refs": [], "hyps": [], "compute": 0.0, "audio": 0.0})
        entry["refs"].append(normalize(text))
        entry["hyps"].append(normalize(hypothesis))
        entry["compute"] += elapsed
        entry["audio"] += len(audio) / 16000
    result = {
        lang: {
            "wer": round(_wer(e["refs"], e["hyps"]), 3),
            "rtf": round(e["compute"] / e["audio"], 3),
        }
        for lang, e in per_lang.items()
    }
    _save("asr", result)
    return result


def bench_tts(config: Config) -> dict:
    """Speed and round-trip intelligibility (TTS → ASR → WER) per voice, plus a listening page."""
    from voicemate.asr.parakeet import ParakeetRecognizer
    from voicemate.audio.codec import resample
    from voicemate.tts.voices import VoiceBank

    voices = VoiceBank.from_config(config)
    asr = ParakeetRecognizer(config.asr.model)
    out_dir = BENCH_DIR / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict] = {}
    rows = []
    for i, (lang, text) in enumerate(corpus()):
        start = time.perf_counter()
        audio, rate = voices.synthesize(text, lang)
        elapsed = time.perf_counter() - start
        wav = out_dir / f"{lang}_{i:02d}.wav"
        sf.write(wav, audio, rate)
        heard = asr.transcribe_text(resample(audio, rate, 16000))
        entry = result.setdefault(lang, {"refs": [], "hyps": [], "times": [], "audio": 0.0})
        entry["refs"].append(normalize(text))
        entry["hyps"].append(normalize(heard))
        entry["times"].append(elapsed)
        entry["audio"] += len(audio) / rate
        rows.append((lang, text, wav.name))
    summary = {
        lang: {
            "voice": config.tts[lang].voice,
            "roundtrip_wer": round(_wer(e["refs"], e["hyps"]), 3),
            "rtf": round(sum(e["times"]) / e["audio"], 3),
            "p50_sentence_ms": round(float(np.median(e["times"])) * 1000),
        }
        for lang, e in result.items()
    }
    items = "\n".join(
        f"<li><b>{lang}</b> {html.escape(text)}<br><audio controls src='{name}'></audio></li>"
        for lang, text, name in rows
    )
    compare = _compare_hungarian_voices(config, out_dir)
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Voice samples</title>"
        "<h1>Voice samples</h1><p>Rate each voice 1-5; the Hungarian voice is set in "
        "<code>[tts.hu] voice</code> of config/voicemate.toml.</p>"
        f"<h2>Hungarian voices side by side</h2>{compare}"
        f"<h2>Configured voices, full corpus</h2><ol>{items}</ol>",
        encoding="utf-8",
    )
    _save("tts", summary)
    return summary


#: Piper voices compared on the listening page.
HU_VOICES: tuple[str, ...] = ("hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium")


def _compare_hungarian_voices(config: Config, out_dir: Path) -> str:
    """Render the first Hungarian sentences with every Piper voice; returns an HTML table."""
    from voicemate import models
    from voicemate.bench import HU_SENTENCES
    from voicemate.tts.normalize import speakable
    from voicemate.tts.piper import PiperSynthesizer

    sentences = HU_SENTENCES[:4]
    header = "".join(f"<th>{voice.split('-')[1]}</th>" for voice in HU_VOICES)
    cells: list[list[str]] = [[] for _ in sentences]
    for voice in HU_VOICES:
        synth = PiperSynthesizer(models.ensure_piper_voice(config, voice))
        for i, text in enumerate(sentences):
            name = f"compare_{voice}_{i}.wav"
            name = config.assistant.name
            spoken = speakable(text, {name: config.assistant.pronunciation.get("hu", name)})
            sf.write(out_dir / name, synth.synthesize(spoken), synth.sample_rate)
            cells[i].append(f"<td><audio controls src='{name}'></audio></td>")
    rows = "".join(
        f"<tr><td>{html.escape(text)}</td>{''.join(row)}</tr>"
        for text, row in zip(sentences, cells, strict=True)
    )
    return f"<table><tr><th>Sentence</th>{header}</tr>{rows}</table>"


#: Scripted tool-choice cases: (prompt, acceptable tools; None means "answer directly").
#: The time is part of every turn's context, so answering it directly is correct.
TOOL_CASES: tuple[tuple[str, tuple[str | None, ...]], ...] = (
    ("Hány óra van most?", ("get_time", None)),
    ("What time is it?", ("get_time", None)),
    ("Milyen idő lesz holnap Gyöngyösön?", ("get_weather",)),
    ("What's the weather tomorrow in Budapest?", ("get_weather",)),
    ("Mennyi 23 szorozva 42-vel?", ("calculator",)),
    ("What is 17% of 340?", ("calculator",)),
    ("Jegyezd fel, hogy holnap hívjam fel az orvost.", ("add_note",)),
    ("Remind me to call the dentist on Monday at 9.", ("add_note",)),
    ("Keress friss cikkeket az arXiv-on a mixture of experts témában.", ("arxiv_search",)),
    ("Find recent arXiv papers on speculative decoding.", ("arxiv_search",)),
    ("Keress gépi tanulási állásokat Budapesten.", ("web_search",)),
    ("Who won the Formula 1 race last weekend?", ("web_search",)),
    ("Szia, hogy vagy?", (None,)),
    ("Explain what a transformer is in one sentence.", (None,)),
)


def bench_llm(config: Config) -> dict:
    """Tool-choice accuracy and first-token latency of the configured Ollama model."""
    import asyncio
    import tempfile

    import httpx
    from langchain_core.messages import HumanMessage, SystemMessage

    from voicemate.agent.graph import make_chat_model
    from voicemate.agent.prompts import system_prompt, turn_context
    from voicemate.agent.tools import ToolContext, build_tools
    from voicemate.agent.tools.search import make_backend
    from voicemate.memory.notes import NotesStore
    from voicemate.memory.store import MemoryStore

    async def run() -> dict:
        from datetime import datetime

        class _NullEmbedder:
            def embed(self, texts: list[str]) -> np.ndarray:
                return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient() as http:
                ctx = ToolContext(
                    config,
                    MemoryStore(Path(tmp) / "m", _NullEmbedder(), config.memory),
                    NotesStore(Path(tmp) / "n.sqlite"),
                    make_backend(config.search, http),
                    http,
                )
                llm = make_chat_model(config.llm).bind_tools(build_tools(ctx))
                correct, rows, ttfts = 0, [], []
                for prompt, expected in TOOL_CASES:
                    lang = "hu" if any(c in prompt for c in "áéíóöőúüű") else "en"
                    context = turn_context(lang, datetime.now().astimezone())
                    messages = [
                        SystemMessage(system_prompt(config)),
                        HumanMessage(f"{context}\n\n{prompt}"),
                    ]
                    start = time.perf_counter()
                    first = None
                    message = None
                    async for chunk in llm.astream(messages):
                        if first is None and (
                            chunk.content or getattr(chunk, "tool_call_chunks", None)
                        ):
                            first = time.perf_counter() - start
                        message = chunk if message is None else message + chunk
                    calls = getattr(message, "tool_calls", []) or []
                    got = calls[0]["name"] if calls else None
                    correct += got in expected
                    ttfts.append(first or 0.0)
                    rows.append({"prompt": prompt, "expected": expected, "got": got})
                ctx.notes.close()
        return {
            "model": config.llm.model,
            "tool_accuracy": f"{correct}/{len(TOOL_CASES)}",
            "ttft_p50_s": round(float(np.median(ttfts)), 3),
            "cases": rows,
        }

    result = asyncio.run(run())
    _save("llm", result)
    return result
