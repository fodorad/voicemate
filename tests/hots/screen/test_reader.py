import unittest

import numpy as np

from tests.helpers import ScriptedScreen
from voicemate.hots.screen.reader import FULL, Frame, ScreenError, TextBox


def frame(*boxes: TextBox) -> Frame:
    return Frame(np.zeros((9, 16, 3), dtype=np.uint8), boxes)


class TestTextBox(unittest.TestCase):
    def test_the_centre_lies_inside_the_box(self):
        box = TextBox("SKY TEMPLE", 1.0, left=0.4, top=0.02, width=0.2, height=0.04)
        self.assertAlmostEqual(box.cx, 0.5)
        self.assertAlmostEqual(box.cy, 0.04)


class TestScriptedScreen(unittest.TestCase):
    def test_frames_play_in_order_and_the_last_one_stays(self):
        first, second = frame(), frame(TextBox("16", 1.0, 0.48, 0.05, 0.03, 0.04))
        screen = ScriptedScreen(first, second)
        self.assertIs(screen.read(), first)
        self.assertIs(screen.read(FULL, fast=True), second)
        self.assertIs(screen.read(), second)
        self.assertEqual(screen.reads[1], (FULL, True))

    def test_a_scripted_error_is_raised_by_the_read(self):
        screen = ScriptedScreen(ScreenError("blocked"))
        with self.assertRaisesRegex(ScreenError, "blocked"):
            screen.read()


if __name__ == "__main__":
    unittest.main()
