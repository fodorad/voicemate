import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from voicemate.hots.screen.log import ScreenLog


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 30, 20, 12, 7)

    def __call__(self) -> datetime:
        return self.now

    def later(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class TestScreenLog(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name) / "game_screens"
        self.clock = Clock()
        self.log = ScreenLog(self.folder, keep=3, clock=self.clock)
        self.image = np.random.default_rng(0).integers(0, 255, (9, 16, 3), dtype=np.uint8)

    def test_screens_are_numbered_inside_a_dated_draft_folder(self):
        draft = self.log.start_draft()
        self.assertEqual(draft.name, "2026-10-30_201207")
        first = self.log.save(self.image, {"phase": "ban"})
        self.clock.later(5)
        second = self.log.save(self.image, {"phase": "pick"})
        self.assertEqual((first.name, second.name), ("01.png", "02.png"))
        summary = json.loads((draft / "draft.json").read_text())
        self.assertEqual([s["phase"] for s in summary["screens"]], ["ban", "pick"])
        self.assertEqual(summary["screens"][1]["at"], "2026-10-30T20:12:12")

    def test_the_saved_picture_is_the_one_given(self):
        from PIL import Image

        path = self.log.save(self.image, {})
        np.testing.assert_array_equal(np.asarray(Image.open(path).convert("RGB")), self.image)

    def test_saving_without_a_draft_starts_one(self):
        path = self.log.save(self.image, {})
        self.assertEqual(path.parent.name, "2026-10-30_201207")

    def test_two_drafts_in_the_same_second_get_different_folders(self):
        self.assertNotEqual(self.log.start_draft(), self.log.start_draft())

    def test_only_the_newest_drafts_are_kept(self):
        for _ in range(5):
            self.log.start_draft()
            self.log.save(self.image, {})
            self.clock.later(60)
        names = sorted(p.name for p in self.folder.iterdir())
        self.assertEqual(names, ["2026-10-30_201407", "2026-10-30_201507", "2026-10-30_201607"])

    def test_foreign_folders_are_never_deleted(self):
        other = self.folder / "notes"
        other.mkdir(parents=True)
        for _ in range(5):
            self.log.start_draft()
            self.clock.later(60)
        self.assertTrue(other.is_dir())

    def test_the_newest_screenshot_is_the_last_of_the_newest_draft(self):
        self.assertIsNone(self.log.newest_screenshot())
        self.log.start_draft()
        self.log.save(self.image, {})
        self.clock.later(60)
        self.log.start_draft()  # an empty newer draft does not hide the older screens
        last = self.log.save(self.image, {})
        self.assertEqual(self.log.newest_screenshot(), last)
        self.clock.later(60)
        self.log.start_draft()
        self.assertEqual(self.log.newest_screenshot(), last)


if __name__ == "__main__":
    unittest.main()
