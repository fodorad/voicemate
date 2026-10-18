"""Spoken and shown reminders.

:class:`Reminders` polls the notes database for reminders whose time has come, marks each as
notified (so it is announced exactly once) and hands it to an announcer supplied by the voice
session, which decides how to show and speak it. The module knows nothing about the event bus
or about speech, so another delivery channel can be added without touching it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from voicemate.memory.notes import Note, NotesStore

logger = logging.getLogger(__name__)

#: Seconds between checks for due reminders.
REMINDER_INTERVAL_S: float = 30.0

#: Spoken before the reminder text, per language (English for any other language).
PREFIXES: dict[str, str] = {"hu": "Emlékeztető: ", "en": "Reminder: "}


def spoken_text(note: Note, lang: str) -> str:
    """What to say for ``note`` in ``lang``."""
    return PREFIXES.get(lang, PREFIXES["en"]) + note.text


class Reminders:
    """Finds due reminders and announces each exactly once.

    Args:
        notes: The notes database.
        announce: Called with every due reminder (after it was marked notified).
        interval_s: Seconds between checks while watching.
        ready: Whether anyone can be told right now; while it is false nothing is consumed,
            so a reminder that falls due with no one listening is announced later.
    """

    def __init__(
        self,
        notes: NotesStore,
        announce: Callable[[Note], Awaitable[None]],
        interval_s: float = REMINDER_INTERVAL_S,
        ready: Callable[[], bool] = lambda: True,
    ) -> None:
        self.notes = notes
        self.announce = announce
        self.interval_s = interval_s
        self.ready = ready

    async def check(self) -> int:
        """Announce the reminders that are due now; returns how many there were."""
        if not self.ready():
            return 0
        due = await asyncio.to_thread(self.notes.due_reminders)
        for note in due:
            await asyncio.to_thread(self.notes.mark_notified, note.id)
            await self.announce(note)
        return len(due)

    async def watch(self) -> None:
        """Check every ``interval_s`` seconds until cancelled; a failed check is logged."""
        while True:
            try:
                await self.check()
            except Exception:
                logger.exception("Reminder check failed")
            await asyncio.sleep(self.interval_s)
