import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from voicemate.memory.notes import NotesStore


class TestNotesStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.notes = NotesStore(Path(self.tmp.name) / "notes.sqlite")

    def tearDown(self):
        self.notes.close()
        self.tmp.cleanup()

    def test_add_list_complete(self):
        first = self.notes.add("buy milk", "todo")
        self.notes.add("idea: voice agent", "note")
        self.assertEqual([n.text for n in self.notes.entries()], ["buy milk", "idea: voice agent"])
        self.assertEqual([n.text for n in self.notes.entries(kind="todo")], ["buy milk"])
        self.assertTrue(self.notes.complete(first.id))
        self.assertEqual([n.text for n in self.notes.entries(kind="todo")], [])
        self.assertEqual(len(self.notes.entries(kind="todo", include_done=True)), 1)
        self.assertFalse(self.notes.complete(999))

    def test_invalid_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            self.notes.add("x", "shopping")

    def test_due_reminders_are_reported_once(self):
        reminder = self.notes.add("call the doctor", "reminder", due="2026-01-01T09:00")
        self.notes.add("later", "reminder", due="2099-01-01T09:00")
        now = datetime(2026, 1, 1, 10, 0)
        due = self.notes.due_reminders(now)
        self.assertEqual([n.id for n in due], [reminder.id])
        self.notes.mark_notified(reminder.id)
        self.assertEqual(self.notes.due_reminders(now), [])

    def test_unparseable_due_date_is_rejected(self):
        with self.assertRaises(ValueError):
            self.notes.add("x", "reminder", due="next tuesday-ish")

    def test_search(self):
        self.notes.add("Kovács doctor phone number", "note")
        self.notes.add("dentist", "todo")
        self.assertEqual(
            [n.text for n in self.notes.search("doctor")], ["Kovács doctor phone number"]
        )

    def test_persistence(self):
        self.notes.add("persist me", "note")
        self.notes.close()
        self.notes = NotesStore(Path(self.tmp.name) / "notes.sqlite")
        self.assertEqual(self.notes.entries()[0].text, "persist me")


if __name__ == "__main__":
    unittest.main()
