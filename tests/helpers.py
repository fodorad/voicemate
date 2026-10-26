"""Deterministic, dependency-free implementations of the model protocols for unit tests.

These are small real implementations (not mocks): they honour the same interfaces as the
production models so the pipeline logic can be exercised without downloading weights.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import numpy as np

#: The ``voicemate`` package directory, for tests that read the source of an architecture rule.
PACKAGE_DIR: Path = Path(__file__).resolve().parents[1] / "voicemate"


def module_source(relative: str) -> str:
    """Source text of a module of the package, e.g. ``"pipeline/orchestrator.py"``."""
    return (PACKAGE_DIR / relative).read_text()


def imported_modules(source: str) -> set[str]:
    """Dotted names of every module a source imports (``import a.b`` and ``from a.b import c``)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def uses_async_syntax(source: str) -> bool:
    """True if the source uses ``async def``, ``await``, ``async for`` or ``async with``."""
    kinds = (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)
    return any(isinstance(node, kinds) for node in ast.walk(ast.parse(source)))


def called_names(source: str) -> set[str]:
    """Names of the functions and methods the source calls (``open(...)``, ``x.write(...)``)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def attributes_of(source: str, owner: str) -> set[str]:
    """Attribute names accessed on something called ``owner``: ``owner.x`` and ``a.owner.x``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            value = node.value
            if (isinstance(value, ast.Name) and value.id == owner) or (
                isinstance(value, ast.Attribute) and value.attr == owner
            ):
                found.add(node.attr)
    return found


class EnergyVAD:
    """RMS-energy speech probability. Deterministic; no model needed.

    Args:
        reference_rms: RMS level mapped to probability 1.0.
        frame_size: Samples per frame.
    """

    def __init__(self, reference_rms: float = 0.05, frame_size: int = 512) -> None:
        self.reference_rms = reference_rms
        self.frame_size = frame_size

    def __call__(self, frame: np.ndarray) -> float:
        """Return ``min(1, rms / reference_rms)``."""
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        return min(1.0, rms / self.reference_rms)

    def reset(self) -> None:
        """Stateless; nothing to reset."""


class ToneSynthesizer:
    """TTS stand-in: 10 ms of a 440 Hz tone per character of input."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.calls: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        n = int(0.01 * self.sample_rate) * len(text)
        t = np.arange(n) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class ScriptedRecognizer:
    """ASR stand-in returning pre-set transcripts in order (then empty strings)."""

    def __init__(self, transcripts: list[str], repeat_last: bool = False) -> None:
        self.transcripts = list(transcripts)
        self.repeat_last = repeat_last
        self.calls = 0

    def transcribe_text(self, audio: np.ndarray) -> str:
        self.calls += 1
        if self.repeat_last and len(self.transcripts) == 1:
            return self.transcripts[0]
        return self.transcripts.pop(0) if self.transcripts else ""


class ScriptedSkill:
    """A skill that answers exactly the texts it was given (a real implementation of the
    :class:`~voicemate.pipeline.skills.Skill` protocol)."""

    def __init__(self, answers: dict[str, str], lang: str = "en") -> None:
        self.answers = answers
        self.lang = lang
        self.asked: list[str] = []

    def answer(self, text: str):
        """The scripted answer for ``text``, or ``None``."""
        from voicemate.pipeline.skills import SkillAnswer

        self.asked.append(text)
        found = self.answers.get(text)
        return None if found is None else SkillAnswer(found, self.lang)


class HashingEmbedder:
    """Bag-of-words feature hashing embedder: identical texts → identical vectors,
    texts sharing words → positive cosine similarity."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                out[row, index] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class StaticSearchBackend:
    """Search backend returning fixed results and counting requests (no network)."""

    name = "static"

    def __init__(self, results=None) -> None:
        from voicemate.agent.tools.search import SearchResult

        self.results = results or [
            SearchResult(
                "MoE survey", "https://example.org/moe", "Mixture of experts routes tokens."
            ),
            SearchResult("Switch Transformer", "https://example.org/switch", "Sparse experts."),
        ]
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int):
        self.queries.append(query)
        return self.results[:max_results]


class StaticArxivBackend:
    """arXiv stand-in returning fixed papers and counting requests (no network)."""

    name = "static-arxiv"

    def __init__(self, papers=None) -> None:
        from voicemate.agent.tools.arxiv_tools import Paper

        self.papers = papers or [
            Paper(
                "2401.00001",
                "Sparse experts at scale",
                ["A. Author", "B. Author"],
                "2024-01-02",
                "We route tokens to experts.",
                "https://arxiv.org/pdf/2401.00001",
            ),
            Paper(
                "2401.00002",
                "Routing without load balancing",
                ["C. Author"],
                "2024-01-03",
                "Balanced routing, no auxiliary loss.",
                "https://arxiv.org/pdf/2401.00002",
            ),
        ]
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int):
        self.queries.append(query)
        return self.papers[:max_results]

    async def get(self, paper_id: str):
        return next((p for p in self.papers if p.arxiv_id == paper_id), None)


def make_tool_context(tmp: str, http=None, search=None, arxiv=None):
    """A ToolContext whose sandbox, memory and notes live under ``tmp``."""
    from datetime import datetime
    from pathlib import Path

    import httpx

    from voicemate.agent.tools.base import ToolContext
    from voicemate.config import Config, MemoryConfig
    from voicemate.memory.notes import NotesStore
    from voicemate.memory.store import MemoryStore

    base = Path(tmp)
    config = Config(data_dir=str(base / "data"))
    config.files.roots = [str(base / "Documents"), str(base / "Downloads")]
    config.files.papers_dir = str(base / "Downloads" / "voicemate-papers")
    for root in config.files.root_paths:
        root.mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(
        base / "data" / "memory",
        HashingEmbedder(),
        MemoryConfig(
            search_cache_similarity=0.9, recall_min_similarity=0.2, fact_min_similarity=0.2
        ),
    )
    from voicemate.agent.tools.research import Research

    client = http or httpx.AsyncClient()
    return ToolContext(
        config=config,
        memory=memory,
        notes=NotesStore(base / "data" / "notes.sqlite"),
        research=Research(
            memory,
            search or StaticSearchBackend(),
            arxiv or StaticArxivBackend(),
            max_results=5,
            http=client,
            # The fetch tests serve pages from a local HTTP server.
            allowed_private_hosts=frozenset({"127.0.0.1"}),
        ),
        http=client,
        now=lambda: datetime(2026, 9, 28, 21, 30).astimezone(),
    )


def make_scripted_chat_model(responses, delay=0.0, token_delay=0.0):
    """Build a :class:`ScriptedChatModel` (a real BaseChatModel) returning ``responses`` in order.

    Each response is a string (plain reply, streamed word by word), an ``AIMessage`` with
    ``tool_calls``, or an exception instance to raise. Every received prompt is recorded in
    ``model.seen``; ``delay`` seconds are awaited before each async response and
    ``token_delay`` seconds before each streamed chunk.
    """
    import asyncio
    import json
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from pydantic import Field

    class ScriptedChatModel(BaseChatModel):
        responses: list[Any] = Field(default_factory=list)
        seen: list[Any] = Field(default_factory=list)
        delay: float = 0.0
        token_delay: float = 0.0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _next(self, messages) -> AIMessage:
            self.seen.append(messages)
            item = self.responses.pop(0) if self.responses else "OK."
            if isinstance(item, Exception):
                raise item
            return AIMessage(item) if isinstance(item, str) else item

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            message = self._next(messages)
            if message.tool_calls:
                chunks = [
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                    for i, c in enumerate(message.tool_calls)
                ]
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_call_chunks=chunks)
                )
                return
            for word in str(message.content).split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            await asyncio.sleep(self.delay)
            for chunk in self._stream(messages, stop, None, **kwargs):
                await asyncio.sleep(self.token_delay)
                yield chunk

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            await asyncio.sleep(self.delay)
            return self._generate(messages, stop, None, **kwargs)

    return ScriptedChatModel(responses=list(responses), delay=delay, token_delay=token_delay)


def tool_call(name, args, call_id="call_1"):
    """An AIMessage requesting one tool call."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        "", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )
