import asyncio
import contextlib
import tempfile
import unittest
from pathlib import Path

from voicemate.memory.notes import Note, NotesStore
from voicemate.pipeline.reminders import Reminders, spoken_text

PAST = "2020-01-01T09:00"
FUTURE = "2999-01-01T09:00"


class RemindersTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.notes = NotesStore(Path(self.tmp.name) / "notes.sqlite")
        self.announced: list[str] = []
        self.still_due_when_announced: list[bool] = []

    def tearDown(self):
        self.notes.close()
        self.tmp.cleanup()

    async def announce(self, note: Note) -> None:
        self.announced.append(note.text)
        self.still_due_when_announced.append(note.id in {n.id for n in self.notes.due_reminders()})

    def reminders(self, announce=None, interval_s=30.0) -> Reminders:
        return Reminders(self.notes, announce or self.announce, interval_s)


class TestCheck(RemindersTestCase):
    async def test_a_due_reminder_is_announced_once(self):
        self.notes.add("call the doctor", "reminder", due=PAST)
        reminders = self.reminders()
        self.assertEqual(await reminders.check(), 1)
        self.assertEqual(self.announced, ["call the doctor"])
        self.assertEqual(await reminders.check(), 0)
        self.assertEqual(self.announced, ["call the doctor"])

    async def test_it_is_marked_notified_before_it_is_announced(self):
        self.notes.add("call the doctor", "reminder", due=PAST)
        await self.reminders().check()
        self.assertEqual(self.still_due_when_announced, [False])

    async def test_nothing_due_announces_nothing(self):
        self.notes.add("later", "reminder", due=FUTURE)
        self.notes.add("a todo", "todo", due=PAST)
        self.notes.add("a note", "note")
        self.assertEqual(await self.reminders().check(), 0)
        self.assertEqual(self.announced, [])

    async def test_every_due_reminder_is_announced_in_due_order(self):
        self.notes.add("second", "reminder", due="2020-01-02T09:00")
        self.notes.add("first", "reminder", due=PAST)
        self.assertEqual(await self.reminders().check(), 2)
        self.assertEqual(self.announced, ["first", "second"])


class TestSpokenText(unittest.TestCase):
    def test_the_prefix_follows_the_language(self):
        note = Note(1, "call Anna", "reminder", PAST, False, "2020-01-01T08:00")
        self.assertEqual(spoken_text(note, "hu"), "Emlékeztető: call Anna")
        self.assertEqual(spoken_text(note, "en"), "Reminder: call Anna")
        self.assertEqual(spoken_text(note, "de"), "Reminder: call Anna")


class TestWatch(RemindersTestCase):
    async def run_watch(self, reminders: Reminders, until) -> None:
        task = asyncio.create_task(reminders.watch())
        for _ in range(200):
            if until():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_the_watcher_keeps_checking(self):
        reminders = self.reminders(interval_s=0.01)
        task = asyncio.create_task(reminders.watch())
        await asyncio.sleep(0.03)
        self.notes.add("added later", "reminder", due=PAST)
        for _ in range(200):
            if self.announced:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.assertEqual(self.announced, ["added later"])

    async def test_a_failing_announcement_does_not_stop_the_watcher(self):
        calls: list[str] = []

        async def flaky(note: Note) -> None:
            calls.append(note.text)
            if note.text == "first":
                raise RuntimeError("speaker unplugged")

        self.notes.add("first", "reminder", due=PAST)
        self.notes.add("second", "reminder", due="2020-01-02T09:00")
        await self.run_watch(self.reminders(flaky, interval_s=0.01), lambda: "second" in calls)
        self.assertEqual(calls, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
