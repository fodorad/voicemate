import os
import sys
import unittest

from voicemate.config import Config

INTEGRATION = bool(os.environ.get("VOICEMATE_INTEGRATION")) and sys.platform == "darwin"
CALIBRATION = Config().data_path / "calibration"


@unittest.skipUnless(INTEGRATION and CALIBRATION.is_dir(), "needs macOS and data/calibration")
class TestRealScreenshots(unittest.TestCase):  # pragma: no cover - integration only
    def test_the_draft_screen_text_is_found_with_positions(self):
        from voicemate.hots.screen.macos import ImageFileReader

        path = sorted(CALIBRATION.glob("*.png"))[5]
        frame = ImageFileReader(lambda: path).read()
        texts = {box.text.upper(): box for box in frame.boxes}
        self.assertIn("SKY TEMPLE", texts)
        self.assertLess(texts["SKY TEMPLE"].top, 0.1)  # the map name is at the top
        self.assertEqual(frame.image.shape[2], 3)

    def test_a_region_limits_the_text_but_not_the_picture(self):
        from voicemate.hots.screen.macos import ImageFileReader

        path = sorted(CALIBRATION.glob("*.png"))[5]
        top = ImageFileReader(lambda: path).read(region=(0.3, 0.0, 0.7, 0.12), fast=True)
        self.assertTrue(top.boxes)
        self.assertTrue(all(box.top < 0.12 for box in top.boxes))
        self.assertGreater(top.image.shape[0], 500)

    def test_a_missing_screenshot_is_reported(self):
        from voicemate.hots.screen.macos import ImageFileReader
        from voicemate.hots.screen.reader import ScreenError

        with self.assertRaises(ScreenError):
            ImageFileReader(lambda: None).read()


if __name__ == "__main__":
    unittest.main()
