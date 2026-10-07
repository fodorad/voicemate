"""Notes, todos and reminders in a small SQLite database."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: Allowed note kinds.
KINDS: tuple[str, ...] = ("note", "todo", "reminder")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    due TEXT,
    done INTEGER NOT NULL DEFAULT 0,
    notified INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL
)
"""


@dataclass
class Note:
    """One note, todo or reminder.

    Attributes:
        id: Primary key.
        text: Content.
        kind: ``note``, ``todo`` or ``reminder``.
        due: ISO-8601 local due time (reminders/todos), if any.
        done: Completed flag.
        created: ISO-8601 creation time.
    """

    id: int
    text: str
    kind: str
    due: str | None
    done: bool
    created: str


class NotesStore:
    """Thread-safe notes database.

    Args:
        path: SQLite file (created if missing).
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row(row: tuple) -> Note:
        return Note(row[0], row[1], row[2], row[3], bool(row[4]), row[5])

    def _query(self, sql: str, params: tuple = ()) -> list[Note]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, kind, due, done, created FROM notes " + sql, params
            ).fetchall()
        return [self._row(r) for r in rows]

    def add(self, text: str, kind: str = "note", due: str | None = None) -> Note:
        """Create a note.

        Args:
            text: Content.
            kind: One of :data:`KINDS`.
            due: ISO-8601 local date/time.

        Raises:
            ValueError: On an unknown kind or unparseable due date.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if due is not None:
            due = datetime.fromisoformat(due).isoformat(timespec="minutes")
        created = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO notes (text, kind, due, created) VALUES (?, ?, ?, ?)",
                (text, kind, due, created),
            )
            self._conn.commit()
            note_id = int(cursor.lastrowid or 0)
        return Note(note_id, text, kind, due, False, created)

    def entries(self, kind: str | None = None, include_done: bool = False) -> list[Note]:
        """Notes in creation order, optionally filtered."""
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if not include_done:
            clauses.append("done = 0")
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        return self._query(where + "ORDER BY id", tuple(params))

    def search(self, text: str) -> list[Note]:
        """Open notes containing ``text`` (case-insensitive)."""
        return self._query("WHERE done = 0 AND text LIKE ? ORDER BY id", (f"%{text}%",))

    def complete(self, note_id: int) -> bool:
        """Mark a note done; returns False if it does not exist."""
        with self._lock:
            cursor = self._conn.execute("UPDATE notes SET done = 1 WHERE id = ?", (note_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def due_reminders(self, now: datetime | None = None) -> list[Note]:
        """Open, not yet notified reminders whose due time has passed."""
        now_iso = (now or datetime.now()).isoformat(timespec="minutes")
        return self._query(
            "WHERE kind = 'reminder' AND done = 0 AND notified = 0 AND due IS NOT NULL "
            "AND due <= ? ORDER BY due",
            (now_iso,),
        )

    def mark_notified(self, note_id: int) -> None:
        """Record that a reminder was announced."""
        with self._lock:
            self._conn.execute("UPDATE notes SET notified = 1 WHERE id = ?", (note_id,))
            self._conn.commit()
