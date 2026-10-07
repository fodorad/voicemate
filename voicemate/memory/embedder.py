"""Text embeddings for long-term memory.

Embeddings run in-process rather than through Ollama: loading an embedding model in Ollama
next to the 22 GB chat model makes Ollama evict the chat model (12 s reload), measured in the
design spikes (docs/sdlc/spec.md §1).
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Maps texts to L2-normalized float32 vectors of shape ``(len(texts), dim)``."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts``."""
        ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model (default ``BAAI/bge-m3``) on MPS when available.

    Calls are serialized with a lock because the model is shared across threads.

    Args:
        model: Hugging Face model id.
        device: ``"cpu"``, ``"mps"`` or ``"auto"`` (MPS when available). CPU is the default:
            the GPU is kept for the LLM, and a query embeds in about 50 ms on the CPU.
    """

    def __init__(self, model: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading embedding model %s on %s", model, device)
        self._model = SentenceTransformer(model, device=device)
        if device == "mps":
            self._model.half()
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` (normalized)."""
        with self._lock:
            vectors = self._model.encode(
                texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
            )
        return np.asarray(vectors, dtype=np.float32)
