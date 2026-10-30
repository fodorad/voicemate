import os
import sys
import unittest

from voicemate.config import load_config

INTEGRATION = bool(os.environ.get("VOICEMATE_INTEGRATION")) and sys.platform == "darwin"
CONFIG = load_config()
CALIBRATION = CONFIG.data_path / "calibration"

#: What Adam's calibration screens show, checked by eye: ``time → (phase, ally, pending, enemy)``.
EXPECTED = {
    "20.12.51": (("ban", True), (), ("auriel", "qhira", "jaina", "artanis"), ()),
    "20.14.51": (
        ("pick", True),
        ("jaina", "johanna"),
        ("auriel", "tychus", "thrall"),
        ("brightwing", "valla", "artanis"),
    ),
    "20.15.29": (
        ("pick", False),
        ("auriel", "jaina", "johanna", "thrall"),
        ("tychus",),
        ("brightwing", "cassia", "dehaka", "valla", "artanis"),
    ),
}


#: Bans shown in the six slots of the screen at 20.15.29, left to right.
BANS = ("xalatath", "abathur", "stitches", "anduin", "qhira", "rehgar")


@unittest.skipUnless(INTEGRATION and CALIBRATION.is_dir(), "needs macOS and data/calibration")
class TestCalibrationScreens(unittest.TestCase):  # pragma: no cover - integration only
    @classmethod
    def setUpClass(cls):
        from voicemate.hots.knowledge import Knowledge

        knowledge = Knowledge(CONFIG.hots_database_path, CONFIG.hots.pages_path)
        cls.roster, cls.maps = knowledge.roster(), knowledge.maps()
        knowledge.close()

    def test_real_screens_read_as_expected(self):
        from voicemate.hots.screen.interpreter import interpret
        from voicemate.hots.screen.macos import ImageFileReader

        for time, (phase, ally, pending, enemy) in EXPECTED.items():
            path = next(CALIBRATION.glob(f"*{time}.png"))
            with self.subTest(screen=time):
                screen = interpret(ImageFileReader(lambda p=path: p).read(), self.roster, self.maps)
                self.assertEqual((screen.phase.kind, screen.phase.mine), phase)
                self.assertEqual(screen.map_slug, "sky-temple")
                self.assertEqual(screen.ally, ally)
                self.assertEqual(screen.pending, pending)
                self.assertEqual(screen.enemy, enemy)

    def test_ban_slots_are_matched_to_portraits(self):
        from voicemate.hots.screen.macos import ImageFileReader
        from voicemate.hots.screen.portraits import Portraits

        portraits = Portraits.from_folder(CONFIG.hots.pages_path.parent / "images")
        path = next(CALIBRATION.glob("*20.15.29.png"))
        self.assertEqual(portraits.bans(ImageFileReader(lambda: path).read(fast=True).image), BANS)
        empty = next(CALIBRATION.glob("*20.12.07.png"))  # the queue screen has no ban slots
        self.assertEqual(portraits.bans(ImageFileReader(lambda: empty).read(fast=True).image), ())


if __name__ == "__main__":
    unittest.main()
