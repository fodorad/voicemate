"""Long-term memory on LanceDB: facts, conversation episodes and the research cache.

Tables (one vector column each, cosine similarity):

* ``facts``: durable facts about the user ("Adam is looking for ML jobs in Budapest").
* ``episodes``: every finished exchange (user text + assistant reply).
* ``searches``: past web/arXiv searches keyed by the *query* embedding, so a repeated question
  is answered from memory instead of hitting the web again.
* ``documents``: text chunks of fetched pages, abstracts and downloaded papers.

Research rows carry a ``ttl_days`` freshness budget; expired rows are ignored by every query.
All methods are synchronous; async callers wrap them with :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.query import LanceVectorQueryBuilder

from voicemate.config import MemoryConfig
from voicemate.memory.embedder import Embedder

#: Seconds per day, used for TTL arithmetic.
DAY: float = 86_400.0


@dataclass
class MemoryHit:
    """One retrieved memory.

    Attributes:
        text: Stored text.
        created: Unix timestamp when it was stored.
        meta: Table-specific extra fields (e.g. ``url``, ``title``, ``thread_id``).
    """

    text: str
    created: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedSearch:
    """A reusable past search.

    Attributes:
        query: The original query.
        results: The stored result list.
        retrieved_at: Unix timestamp of the original web request.
    """

    query: str
    results: list[dict[str, Any]]
    retrieved_at: float


@dataclass
class Recall:
    """Everything memory knows that is relevant to one user message.

    Attributes:
        facts: Matching user facts.
        episodes: Matching past exchanges.
        documents: Matching research chunks.
    """

    facts: list[MemoryHit] = field(default_factory=list)
    episodes: list[MemoryHit] = field(default_factory=list)
    documents: list[MemoryHit] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when nothing relevant was found."""
        return not (self.facts or self.episodes or self.documents)


def _sql_str(value: str) -> str:
    """Quote a string literal for a LanceDB ``where`` clause."""
    return "'" + value.replace("'", "''") + "'"


class _Index:
    """The database, embedder and similarity search shared by the concept interfaces."""

    def __init__(
        self, path: Path, embedder: Embedder, config: MemoryConfig, clock: Callable[[], float]
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.config = config
        self.clock = clock
        self._db = lancedb.connect(str(path))
        dim = int(embedder.embed(["dimension probe"]).shape[1])
        vector = pa.field("vector", pa.list_(pa.float32(), dim))
        text, num = pa.string(), pa.float64()
        schemas = {
            "facts": [("id", text), ("text", text), ("created", num)],
            "episodes": [
                ("id", text),
                ("text", text),
                ("user", text),
                ("assistant", text),
                ("lang", text),
                ("thread_id", text),
                ("created", num),
            ],
            "searches": [
                ("id", text),
                ("text", text),
                ("kind", text),
                ("results", text),
                ("created", num),
                ("ttl_days", num),
            ],
            "documents": [
                ("id", text),
                ("text", text),
                ("url", text),
                ("title", text),
                ("source", text),
                ("created", num),
                ("ttl_days", num),
            ],
        }
        self.tables = {
            name: self._db.create_table(
                name,
                schema=pa.schema([pa.field(n, t) for n, t in columns] + [vector]),
                exist_ok=True,
            )
            for name, columns in schemas.items()
        }

    def vector(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]

    def fresh(self) -> str:
        return f"created + ttl_days * {DAY} >= {self.clock()}"

    def search(
        self,
        table: str,
        vector: np.ndarray,
        limit: int,
        min_similarity: float,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        builder = cast(LanceVectorQueryBuilder, self.tables[table].search(vector.tolist()))
        query = builder.distance_type("cosine").limit(limit)
        if where:
            query = query.where(where, prefilter=True)
        rows = []
        for row in query.to_list():
            row["similarity"] = 1.0 - float(row.pop("_distance"))
            row.pop("vector", None)
            if row["similarity"] >= min_similarity:
                rows.append(row)
        return rows

    @staticmethod
    def hit(row: dict[str, Any]) -> MemoryHit:
        meta = {k: v for k, v in row.items() if k not in ("text", "similarity", "created")}
        return MemoryHit(row["text"], row["created"], meta)


class Facts:
    """Durable facts about the user ("Adam is looking for ML jobs in Budapest")."""

    def __init__(self, index: _Index) -> None:
        self._index = index

    def add(self, text: str) -> bool:
        """Store a fact, replacing a near-duplicate.

        Returns:
            True when an existing fact was replaced.
        """
        ix = self._index
        vector = ix.vector(text)
        duplicates = ix.search("facts", vector, 1, ix.config.fact_dedupe_similarity)
        for row in duplicates:
            ix.tables["facts"].delete(f"id = {_sql_str(row['id'])}")
        row = {"id": uuid.uuid4().hex, "text": text, "created": ix.clock(), "vector": vector}
        ix.tables["facts"].add([row])
        return bool(duplicates)

    def search(self, query: str, limit: int = 5) -> list[MemoryHit]:
        """Facts relevant to ``query``."""
        ix = self._index
        rows = ix.search("facts", ix.vector(query), limit, ix.config.fact_min_similarity)
        return [ix.hit(r) for r in rows]


class Episodes:
    """Finished exchanges between the user and the assistant."""

    def __init__(self, index: _Index) -> None:
        self._index = index

    def add(self, user: str, assistant: str, lang: str, thread_id: str) -> None:
        """Store one finished exchange."""
        ix = self._index
        text = f"User: {user}\nAssistant: {assistant}"
        row = {
            "id": uuid.uuid4().hex,
            "text": text,
            "user": user,
            "assistant": assistant,
            "lang": lang,
            "thread_id": thread_id,
            "created": ix.clock(),
            "vector": ix.vector(text),
        }
        ix.tables["episodes"].add([row])

    def search(self, query: str, limit: int = 3) -> list[MemoryHit]:
        """Past exchanges relevant to ``query``."""
        ix = self._index
        rows = ix.search("episodes", ix.vector(query), limit, ix.config.recall_min_similarity)
        return [ix.hit(r) for r in rows]


class ResearchMemory:
    """Stored searches and the text chunks of pages, abstracts and papers."""

    def __init__(self, index: _Index) -> None:
        self._index = index

    def store_search(
        self, query: str, kind: str, results: list[dict[str, Any]], ttl_days: float
    ) -> None:
        """Remember the results of a web or arXiv search."""
        ix = self._index
        row = {
            "id": uuid.uuid4().hex,
            "text": query,
            "kind": kind,
            "results": json.dumps(results, ensure_ascii=False),
            "created": ix.clock(),
            "ttl_days": float(ttl_days),
            "vector": ix.vector(query),
        }
        ix.tables["searches"].add([row])

    def cached_search(self, query: str, kind: str) -> CachedSearch | None:
        """A fresh stored search whose query is similar enough to ``query``, if any."""
        ix = self._index
        rows = ix.search(
            "searches",
            ix.vector(query),
            1,
            ix.config.search_cache_similarity,
            where=f"kind = {_sql_str(kind)} AND {ix.fresh()}",
        )
        if not rows:
            return None
        row = rows[0]
        return CachedSearch(row["text"], json.loads(row["results"]), row["created"])

    def add_documents(
        self, chunks: list[str], url: str, title: str, source: str, ttl_days: float
    ) -> None:
        """Store text chunks of one document (page, abstract or paper)."""
        if not chunks:
            return
        ix = self._index
        vectors = ix.embedder.embed(chunks)
        now = ix.clock()
        document = uuid.uuid4().hex
        ix.tables["documents"].add(
            [
                {
                    # "<document>-<position>": sorting by id restores the chunk order.
                    "id": f"{document}-{position:05d}",
                    "text": chunk,
                    "url": url,
                    "title": title,
                    "source": source,
                    "created": now,
                    "ttl_days": float(ttl_days),
                    "vector": vector,
                }
                for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ]
        )

    def documents_for_url(self, url: str) -> list[MemoryHit]:
        """Fresh stored chunks of ``url`` in document order (newest copy first)."""
        ix = self._index
        rows = (
            ix.tables["documents"]
            .search()
            .where(f"url = {_sql_str(url)} AND {ix.fresh()}")
            .limit(10_000)
            .to_list()
        )
        rows.sort(key=lambda r: (-r["created"], r["id"]))
        return [
            MemoryHit(r["text"], r["created"], {"url": r["url"], "title": r["title"]}) for r in rows
        ]


class MemoryStore:
    """LanceDB-backed memory: one database, three narrow interfaces.

    Callers use only the slice they need: :attr:`facts`, :attr:`episodes` or :attr:`research`;
    :meth:`recall` combines them for the agent.

    Args:
        path: Database directory.
        embedder: Text embedder.
        config: Similarity thresholds.
        clock: Time source (injectable for tests).
    """

    def __init__(
        self,
        path: Path,
        embedder: Embedder,
        config: MemoryConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._index = _Index(path, embedder, config, clock)
        self.facts = Facts(self._index)
        self.episodes = Episodes(self._index)
        self.research = ResearchMemory(self._index)

    def recall(self, query: str) -> Recall:
        """Facts, episodes and research relevant to ``query`` (one embedding pass)."""
        ix = self._index
        vector = ix.vector(query)
        cfg = ix.config
        return Recall(
            facts=[ix.hit(r) for r in ix.search("facts", vector, 5, cfg.fact_min_similarity)],
            episodes=[
                ix.hit(r) for r in ix.search("episodes", vector, 3, cfg.recall_min_similarity)
            ],
            documents=[
                ix.hit(r)
                for r in ix.search(
                    "documents", vector, 4, cfg.recall_min_similarity, where=ix.fresh()
                )
            ],
        )
