"""Split streamed LLM text into speakable chunks for incremental TTS.

A chunk is emitted as soon as a sentence boundary is certain, so the first sentence is
spoken while the model is still generating the rest. The very first chunk of a turn may be
cut earlier at a clause boundary (comma, semicolon, colon) to reduce time-to-first-audio.
"""

from __future__ import annotations

import re

#: Abbreviations (lower-case, without the dot) that never end a sentence.
ABBREVIATIONS: frozenset[str] = frozenset(
    """pl stb kb ill vö dr ifj id özv ún sz u krt mr mrs ms prof st vs etc approx dept e.g i.e
    no jan feb márc ápr jún júl aug szept okt nov dec""".split()
)

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:](?=\s)")


class SpeechChunker:
    """Incremental sentence splitter.

    Args:
        min_first_chars: The first chunk of a turn may end at a clause boundary once it is
            at least this long.
        max_chars: Chunks longer than this are cut at the last space.
    """

    def __init__(self, min_first_chars: int = 40, max_chars: int = 250) -> None:
        self.min_first_chars = min_first_chars
        self.max_chars = max_chars
        self.reset()

    def reset(self) -> None:
        """Forget buffered text; the next chunk counts as the first of a new turn."""
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        """Add streamed text and return any chunks that are now complete."""
        self._buffer += text
        chunks: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:].lstrip()
            if chunk:
                chunks.append(chunk)
                self._emitted += 1
        return chunks

    def flush(self) -> list[str]:
        """Return the remaining text as final chunk(s) at the end of a turn."""
        rest = self._buffer.strip()
        self._buffer = ""
        chunks: list[str] = []
        while len(rest) > self.max_chars:
            cut = rest.rfind(" ", 0, self.max_chars)
            cut = cut if cut > 0 else self.max_chars
            chunks.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            chunks.append(rest)
        self._emitted += len(chunks)
        return chunks

    # ── boundary detection ────────────────────────────────────────────────────

    def _find_cut(self) -> int | None:
        buffer = self._buffer
        newline = buffer.find("\n")
        for match in _SENTENCE_END.finditer(buffer):
            if newline != -1 and newline < match.start():
                break
            end = match.end()
            following = buffer[end:].lstrip()
            if not following:
                return None  # cannot decide until the next word arrives
            if self._is_boundary(buffer[: match.start()], match.group(), following[0]):
                return end
        if newline != -1:
            return newline + 1
        if self._emitted == 0 and len(buffer) >= self.min_first_chars:
            clauses = [m.end() for m in _CLAUSE_END.finditer(buffer)]
            usable = [c for c in clauses if c >= self.min_first_chars]
            if usable:
                return usable[0]
        if len(buffer) > self.max_chars:
            cut = buffer.rfind(" ", 0, self.max_chars)
            return cut if cut > 0 else self.max_chars
        return None

    @staticmethod
    def _is_boundary(before: str, punctuation: str, next_char: str) -> bool:
        if not punctuation.startswith("."):
            return True
        if next_char.islower():
            return False  # "pl. a", "2. helyen", "e.g. with"
        word = before.rsplit(None, 1)[-1].lower() if before.strip() else ""
        word = word.strip("(\"'")
        if word in ABBREVIATIONS:
            return False
        return not (len(word) == 1 and word.isalpha())  # initials such as "J. Smith"
