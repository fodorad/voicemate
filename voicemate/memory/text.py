"""Text utilities for the research memory: chunking and freshness classification."""

from __future__ import annotations

import re

#: Default chunk size (characters) for research documents.
CHUNK_CHARS: int = 1200

#: Words that mark a query as time-sensitive (answers go stale within a day).
TIME_SENSITIVE: frozenset[str] = frozenset(
    """news hír hírek today ma mai latest legújabb legfrissebb current jelenlegi job jobs állás
    állások price ár árfolyam weather időjárás score eredmény breaking now most""".split()
)

#: Freshness budget in days per source kind (``web`` may be shortened to 1 day).
TTL_DAYS: dict[str, int] = {"web": 30, "page": 30, "arxiv": 365, "paper": 365}

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def ttl_days_for(query: str, source: str) -> int:
    """How long a research result stays fresh.

    Args:
        query: The search query (time-sensitive words shorten web freshness to 1 day).
        source: ``web``, ``page``, ``arxiv`` or ``paper``.
    """
    if source in ("web", "page"):
        words = set(re.findall(r"\w+", query.lower()))
        if words & TIME_SENSITIVE:
            return 1
    return TTL_DAYS.get(source, 30)


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(paragraph):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most ``max_chars``, preferring paragraph and sentence ends.

    Args:
        text: Document text.
        max_chars: Maximum chunk length.

    Returns:
        Non-empty chunks in document order.
    """
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    chunks: list[str] = []
    current = ""
    for paragraph in filter(None, paragraphs):
        for piece in _split_long(paragraph, max_chars):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks
