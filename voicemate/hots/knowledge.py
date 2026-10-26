"""The local HotS knowledge base, filled from the saved Icy Veins pages.

:class:`Knowledge` keeps one SQLite file. :meth:`Knowledge.refresh` compares the pages folder
with what was imported last time (file size and modification time) and re-imports only the
pages that are new or changed, so starting Ava costs almost nothing once the pages are known.
Ava never fetches the pages: Adam saves them from his browser, and a changed page shows up in
the :class:`RefreshReport` so he knows what he re-saved.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from voicemate.hots import pages
from voicemate.hots.roster import Hero, Roster

logger = logging.getLogger(__name__)

#: Tier-list pages that are not about one map.
GENERAL_LISTS: frozenset[str] = frozenset({"general", "master", "aram", "quick-match"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
    page TEXT PRIMARY KEY, kind TEXT, mtime_ns INTEGER, size INTEGER,
    modified TEXT, imported_at REAL);
CREATE TABLE IF NOT EXISTS heroes (
    slug TEXT PRIMARY KEY, name TEXT, role TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS relations (
    hero TEXT, other TEXT, kind TEXT, rank INTEGER, source TEXT);
CREATE INDEX IF NOT EXISTS relations_hero ON relations (hero, kind, rank);
CREATE INDEX IF NOT EXISTS relations_other ON relations (other, kind);
CREATE TABLE IF NOT EXISTS map_ratings (
    hero TEXT, map TEXT, rating TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS map_ratings_hero ON map_ratings (hero, map);
CREATE TABLE IF NOT EXISTS tiers (
    list_key TEXT, hero TEXT, tier TEXT, role TEXT, ban INTEGER, source TEXT);
CREATE INDEX IF NOT EXISTS tiers_list ON tiers (list_key, hero);
CREATE TABLE IF NOT EXISTS maps (slug TEXT PRIMARY KEY, name TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    source TEXT, kind TEXT, heading TEXT, text TEXT);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks (source);
"""

_ROW_TABLES = ("heroes", "relations", "map_ratings", "tiers", "maps", "chunks")

_GUIDE = re.compile(r"^(?P<slug>.+)-build-guide$")
_TIERS = re.compile(r"^heroes-of-the-storm-(?P<key>.+)-tier-list$")
_MAP = re.compile(r"^(?P<slug>.+)-map-guide$")
_TEXT = re.compile(r"^(?P<slug>.+)-(?P<kind>abilities-strategy|talents)$")


@dataclass(frozen=True)
class PageId:
    """What a saved file is.

    Attributes:
        kind: ``guide``, ``tiers``, ``map``, ``abilities-strategy`` or ``talents``.
        key: Hero slug, tier-list key or map slug.
    """

    kind: str
    key: str


def classify(stem: str) -> PageId | None:
    """The kind of page a file stem (``heroes_`` prefix removed) is, or ``None`` to ignore it."""
    for pattern, kind in ((_GUIDE, "guide"), (_TIERS, "tiers"), (_MAP, "map")):
        match = pattern.match(stem)
        if match:
            return PageId(kind, match.group(match.lastindex or 1))
    text = _TEXT.match(stem)
    return PageId(text.group("kind"), text.group("slug")) if text else None


@dataclass(frozen=True)
class TierRank:
    """A hero's place in one tier list."""

    tier: str
    role: str
    ban: bool


@dataclass(frozen=True)
class RefreshReport:
    """What :meth:`Knowledge.refresh` did.

    Attributes:
        added: Pages imported for the first time.
        updated: Pages whose file changed and were imported again.
        removed: Pages no longer in the folder (their rows were dropped).
        failed: ``(page, reason)`` for pages that could not be parsed (old rows are kept).
        unchanged: Number of pages left alone.
    """

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        """Whether anything was added, updated, removed or failed."""
        return bool(self.added or self.updated or self.removed or self.failed)

    def summary(self) -> str:
        """One line for logs and notifications; empty when nothing changed."""
        parts = []
        for label, items in (("new", self.added), ("updated", self.updated)):
            if items:
                parts.append(
                    f"{len(items)} {label} ({', '.join(items[:5])}{'…' if len(items) > 5 else ''})"
                )
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.failed:
            parts.append(
                f"{len(self.failed)} unreadable ({', '.join(p for p, _ in self.failed[:3])})"
            )
        return "HotS pages: " + ", ".join(parts) if parts else ""


@dataclass
class _Parsed:
    """The rows one page contributes."""

    heroes: list[tuple[str, str, str]] = field(default_factory=list)
    relations: list[tuple[str, str, str, int]] = field(default_factory=list)
    map_ratings: list[tuple[str, str, str]] = field(default_factory=list)
    tiers: list[tuple[str, str, str, str, int]] = field(default_factory=list)
    maps: list[tuple[str, str]] = field(default_factory=list)
    chunks: list[tuple[str, str, str]] = field(default_factory=list)
    modified: str = ""


def _parse(page_id: PageId, html: str) -> _Parsed:
    parsed = _Parsed()
    if page_id.kind == "guide":
        guide = pages.parse_guide(html, page_id.key)
        parsed.modified = guide.modified
        parsed.heroes.append((guide.slug, guide.name, guide.role))
        for rank, other in enumerate(guide.counters):
            parsed.relations.append((guide.slug, other, "counter", rank))
        for rank, other in enumerate(guide.synergies):
            parsed.relations.append((guide.slug, other, "synergy", rank))
        parsed.map_ratings = [(guide.slug, m, r) for m, r in guide.maps.items()]
        for heading, items in (("Strengths", guide.strengths), ("Weaknesses", guide.weaknesses)):
            if items:
                parsed.chunks.append((f"{guide.name} {heading.lower()}", heading, "; ".join(items)))
        for heading, text in (("Synergies", guide.synergy_text), ("Counters", guide.counter_text)):
            if text:
                parsed.chunks.append((f"{guide.name} {heading.lower()}", heading, text))
    elif page_id.kind == "tiers":
        tier_list = pages.parse_tier_list(html)
        parsed.modified = tier_list.modified
        parsed.tiers = [
            (page_id.key, e.hero, e.tier, e.role, int(e.ban)) for e in tier_list.entries
        ]
    else:
        text_page = pages.parse_text_page(html)
        parsed.modified = text_page.modified
        if page_id.kind == "map":
            parsed.maps.append((page_id.key, text_page.title.removesuffix(" Map Guide")))
        parsed.chunks = [(text_page.title, h, t) for h, t in text_page.sections]
    return parsed


class Knowledge:
    """Heroes, counters, synergies, map ratings and tier lists from the saved pages.

    Args:
        database: SQLite file (created when missing).
        pages_dir: Folder with the saved ``heroes_*.html`` pages.
    """

    def __init__(self, database: Path, pages_dir: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.pages_dir = pages_dir
        self._db = sqlite3.connect(database, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the database."""
        with self._lock:
            self._db.close()

    # ── import ───────────────────────────────────────────────────────────────

    def refresh(self) -> RefreshReport:
        """Import the pages that are new or changed and drop the ones that disappeared."""
        found: dict[str, tuple[PageId, Path]] = {}
        if self.pages_dir.exists():
            for path in sorted(self.pages_dir.glob("heroes_*.html")):
                stem = path.stem.removeprefix("heroes_")
                page_id = classify(stem)
                if page_id is not None:
                    found[stem] = (page_id, path)
        with self._lock:
            known = {
                row[0]: (row[1], row[2])
                for row in self._db.execute("SELECT page, mtime_ns, size FROM manifest")
            }
        added, updated, failed = [], [], []
        unchanged = 0
        for stem, (page_id, path) in found.items():
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if known.get(stem) == signature:
                unchanged += 1
                continue
            try:
                parsed = _parse(page_id, path.read_text(encoding="utf-8"))
            except (pages.PageError, OSError, UnicodeDecodeError) as exc:
                logger.warning("HotS page %s skipped: %s", stem, exc)
                failed.append((stem, str(exc)))
                continue
            self._store(stem, page_id, signature, parsed)
            (updated if stem in known else added).append(stem)
        removed = sorted(set(known) - set(found))
        for stem in removed:
            self._forget(stem)
        report = RefreshReport(
            tuple(added), tuple(updated), tuple(removed), tuple(failed), unchanged
        )
        if report.changed:
            logger.info(report.summary())
        return report

    def _forget(self, stem: str) -> None:
        with self._lock, self._db:
            for table in _ROW_TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE source = ?", (stem,))  # noqa: S608
            self._db.execute("DELETE FROM manifest WHERE page = ?", (stem,))

    def _store(
        self, stem: str, page_id: PageId, signature: tuple[int, int], parsed: _Parsed
    ) -> None:
        with self._lock, self._db:
            for table in _ROW_TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE source = ?", (stem,))  # noqa: S608
            db = self._db
            db.executemany(
                "INSERT OR REPLACE INTO heroes VALUES (?, ?, ?, ?)",
                [(*row, stem) for row in parsed.heroes],
            )
            db.executemany(
                "INSERT INTO relations VALUES (?, ?, ?, ?, ?)",
                [(*row, stem) for row in parsed.relations],
            )
            db.executemany(
                "INSERT INTO map_ratings VALUES (?, ?, ?, ?)",
                [(*row, stem) for row in parsed.map_ratings],
            )
            db.executemany(
                "INSERT INTO tiers VALUES (?, ?, ?, ?, ?, ?)",
                [(*row, stem) for row in parsed.tiers],
            )
            db.executemany(
                "INSERT OR REPLACE INTO maps VALUES (?, ?, ?)",
                [(*row, stem) for row in parsed.maps],
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                [(stem, page_id.kind, heading, text) for _, heading, text in parsed.chunks],
            )
            db.execute(
                "INSERT OR REPLACE INTO manifest VALUES (?, ?, ?, ?, ?, ?)",
                (stem, page_id.kind, *signature, parsed.modified, time.time()),
            )

    # ── queries ──────────────────────────────────────────────────────────────

    def _all(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def roster(self) -> Roster:
        """All known heroes."""
        return Roster(Hero(*row) for row in self._all("SELECT slug, name, role FROM heroes"))

    def counters_of(self, hero: str) -> tuple[str, ...]:
        """Heroes that counter ``hero``, in the page's order."""
        return self._related(hero, "counter")

    def synergies_of(self, hero: str) -> tuple[str, ...]:
        """Heroes that work well with ``hero``, in the page's order."""
        return self._related(hero, "synergy")

    def countered_by(self, hero: str) -> tuple[str, ...]:
        """Heroes that ``hero`` counters (their pages list ``hero`` as a counter)."""
        rows = self._all("SELECT hero FROM relations WHERE other = ? AND kind = 'counter'", (hero,))
        return tuple(row[0] for row in rows)

    def _related(self, hero: str, kind: str) -> tuple[str, ...]:
        rows = self._all(
            "SELECT other FROM relations WHERE hero = ? AND kind = ? ORDER BY rank", (hero, kind)
        )
        return tuple(row[0] for row in rows)

    def map_rating(self, hero: str, map_slug: str) -> str | None:
        """``stronger``, ``average`` or ``weaker`` for ``hero`` on the map, if known."""
        rows = self._all(
            "SELECT rating FROM map_ratings WHERE hero = ? AND map = ?", (hero, map_slug)
        )
        return rows[0][0] if rows else None

    def tier(self, hero: str, list_key: str) -> TierRank | None:
        """The hero's place in a tier list (``general``, ``master``, a map slug, ...)."""
        rows = self._all(
            "SELECT tier, role, ban FROM tiers WHERE list_key = ? AND hero = ?", (list_key, hero)
        )
        return TierRank(rows[0][0], rows[0][1], bool(rows[0][2])) if rows else None

    def maps(self) -> dict[str, str]:
        """Map slug → map name."""
        return dict(self._all("SELECT slug, name FROM maps ORDER BY name"))

    def chunks(self, source_kind: str | None = None) -> list[tuple[str, str, str]]:
        """Stored text sections as ``(page, heading, text)``, optionally of one page kind."""
        if source_kind is None:
            rows = self._all("SELECT source, heading, text FROM chunks")
        else:
            rows = self._all(
                "SELECT source, heading, text FROM chunks WHERE kind = ?", (source_kind,)
            )
        return rows
