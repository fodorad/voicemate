"""Process-wide runtime: every model and store, loaded once and shared by all sessions."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx

from voicemate.agent.model_select import ModelChoice
from voicemate.asr.base import SpeechRecognizer
from voicemate.asr.langid import LanguageDetector
from voicemate.audio.vad import SpeechProbability
from voicemate.config import Config
from voicemate.memory.notes import NotesStore
from voicemate.memory.store import MemoryStore
from voicemate.tts.voices import VoiceBank

logger = logging.getLogger(__name__)


def use_offline_hub_if_cached(repos: list[str]) -> bool:
    """Switch Hugging Face Hub to offline mode when every repository is already cached.

    Loading a cached model still sends metadata requests to huggingface.co; for a private
    assistant nothing but explicit tool calls should leave the machine.

    Returns:
        True when offline mode was enabled.
    """
    from huggingface_hub import snapshot_download

    for repo in repos:
        try:
            snapshot_download(repo, local_files_only=True)
        except Exception:
            logger.info("%s is not cached yet; Hugging Face Hub stays online", repo)
            return False
    # The environment variable covers libraries imported later; huggingface_hub itself read it
    # at import time already, so its module constant is switched as well.
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import constants

    constants.HF_HUB_OFFLINE = True
    return True


@dataclass
class Runtime:
    """Shared components.

    Attributes:
        config: Application configuration.
        recognizer: Speech recognizer (``None`` for text-only use).
        voices: TTS voices (``None`` for text-only use).
        vad_factory: Creates a fresh VAD model per session (VAD is stateful).
        graph: Compiled LangGraph agent.
        model_choice: The chat model chosen by the memory guard (None in tests).
        memory: Long-term memory store.
        notes: Notes database.
        langid: Language detector.
        asr_executor: Single worker thread for ASR (MLX is not re-entrant).
        tts_executor: Single worker thread for TTS.
    """

    config: Config
    recognizer: SpeechRecognizer | None
    voices: VoiceBank | None
    vad_factory: Callable[[], SpeechProbability]
    graph: Any
    memory: MemoryStore | None
    notes: NotesStore | None
    langid: LanguageDetector = field(default_factory=LanguageDetector)
    model_choice: ModelChoice | None = None
    asr_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(1, thread_name_prefix="asr")
    )
    tts_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(1, thread_name_prefix="tts")
    )
    _stack: AsyncExitStack | None = None

    async def aclose(self) -> None:
        """Release executors, the HTTP client, the checkpointer and databases."""
        if self._stack is not None:
            await self._stack.aclose()
        self.asr_executor.shutdown(wait=False, cancel_futures=True)
        self.tts_executor.shutdown(wait=False, cancel_futures=True)
        if self.notes is not None:
            self.notes.close()

    @classmethod
    async def load(
        cls, config: Config, speech: bool = True
    ) -> Runtime:  # pragma: no cover - loads real models
        """Load every model and open every store.

        Args:
            config: Application configuration.
            speech: Load ASR and TTS (False for the text-only ``chat`` command).
        """
        import asyncio

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from voicemate.agent.graph import build_graph, make_chat_model, warm_up
        from voicemate.agent.model_select import probe_and_choose
        from voicemate.agent.tools import ToolContext, build_tools
        from voicemate.agent.tools.base import USER_AGENT
        from voicemate.agent.tools.search import make_backend
        from voicemate.audio.vad import SileroVAD
        from voicemate.memory.embedder import SentenceTransformerEmbedder

        use_offline_hub_if_cached([config.asr.model, config.memory.embed_model])
        data = config.data_path
        data.mkdir(parents=True, exist_ok=True)
        stack = AsyncExitStack()
        http = await stack.enter_async_context(
            httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT})
        )
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(data / "checkpoints.sqlite"))
        )
        embedder = await asyncio.to_thread(
            SentenceTransformerEmbedder, config.memory.embed_model, config.memory.embed_device
        )
        memory = await asyncio.to_thread(MemoryStore, data / "memory", embedder, config.memory)
        notes = NotesStore(data / "notes.sqlite")
        recognizer = voices = None
        if speech:
            from voicemate.asr.parakeet import ParakeetRecognizer

            voices = await asyncio.to_thread(VoiceBank.from_config, config)
            recognizer = await asyncio.to_thread(
                ParakeetRecognizer, config.asr.model, config.asr.device
            )
        ctx = ToolContext(config, memory, notes, make_backend(config.search, http), http)
        try:
            choice = await asyncio.to_thread(probe_and_choose, config.llm)
        except Exception as exc:  # Ollama down: keep the preferred model, turns report errors
            logger.error("Cannot query Ollama (%s). Is Ollama running?", exc)
            choice = ModelChoice(config.llm.profile, config.llm.model, False, False, str(exc))
        llm, tools = make_chat_model(config.llm, choice.model), build_tools(ctx)
        graph = build_graph(llm, tools, config, memory, checkpointer)
        logger.info("Warming up %s (loads the model and caches the prompt prefix)", choice.model)
        try:
            await warm_up(llm, tools, config)
        except Exception as exc:  # Ollama down: the UI still starts and reports errors per turn
            logger.error("LLM warm-up failed: %s. Is Ollama running?", exc)
        logger.info("Runtime ready (LLM %s)", choice.model)
        return cls(
            config=config,
            langid=LanguageDetector(config.assistant.default_language),
            model_choice=choice,
            recognizer=recognizer,
            voices=voices,
            vad_factory=SileroVAD,
            graph=graph,
            memory=memory,
            notes=notes,
            _stack=stack,
        )
