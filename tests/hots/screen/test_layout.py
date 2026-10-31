import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from voicemate.hots.screen.layout import (
    BROWSE_AT,
    MAP_AT,
    REFERENCE,
    Calibration,
    Layout,
    measure,
)
from voicemate.hots.screen.reader import Frame, TextBox

MAPS = {"sky-temple": "Sky Temple", "cursed-hollow": "Cursed Hollow"}


def shown(layout: Layout, *extra: tuple[str, float, float], noise: float = 0.0) -> Frame:
    """What a screen with this layout shows: the two anchor texts, a marker and ``extra`` texts.

    ``extra`` texts are ``(text, x, y)`` centres on the reference canvas.
    """
    width, height = layout.size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    boxes = []
    for text, x, y in (("SKY TEMPLE", *MAP_AT), ("Browse All Heroes", *BROWSE_AT), *extra):
        cx = layout.center + (x - REFERENCE[0] / 2) * layout.scale
        cy = layout.top + y * layout.scale + noise
        w, h = 100 * layout.scale, 20 * layout.scale
        boxes.append(
            TextBox(text, 1.0, (cx - w / 2) / width, (cy - h / 2) / height, w / width, h / height)
        )
    return Frame(image, tuple(boxes))


def place(frame: Frame, layout: Layout) -> Frame:
    """A canvas-sized ``frame`` as a machine with this ``layout`` would capture it."""
    scale, (width, height) = layout.scale, layout.size
    side = round(REFERENCE[0] * scale), round(REFERENCE[1] * scale)
    canvas = Image.fromarray(frame.image).resize(side, Image.Resampling.NEAREST)
    picture = Image.new("RGB", layout.size)
    left = layout.center - REFERENCE[0] / 2 * scale
    picture.paste(canvas, (round(left), round(layout.top)))
    boxes = tuple(
        TextBox(
            b.text,
            b.confidence,
            (left + b.left * REFERENCE[0] * scale) / width,
            (layout.top + b.top * REFERENCE[1] * scale) / height,
            b.width * REFERENCE[0] * scale / width,
            b.height * REFERENCE[1] * scale / height,
        )
        for b in frame.boxes
    )
    return Frame(np.asarray(picture), boxes)


class TestMeasure(unittest.TestCase):
    def test_the_layout_is_recovered_from_the_two_anchor_texts(self):
        for layout in (
            Layout((1920, 1080), 1.0, 960.0, 0.0),
            Layout((3840, 2160), 2.0, 1920.0, 0.0),
            Layout((1600, 940), 900 / 1080, 800.0, 40.0),  # a window under a title bar
            Layout((2560, 1080), 1.0, 1280.0, 0.0),  # wider than 16:9
            Layout((1680, 1050), 0.875, 840.0, 52.0),  # 16:10 with bars
        ):
            with self.subTest(size=layout.size):
                found = measure(shown(layout), MAPS)
                self.assertIsNotNone(found)
                self.assertTrue(found.same_as(layout), found)

    def test_without_both_anchors_nothing_is_measured(self):
        layout = Layout.default((1920, 1080))
        frame = shown(layout)
        for keep in (1, 0):  # only the map name, only "Browse All Heroes"
            with self.subTest(kept=keep):
                lonely = Frame(frame.image, (frame.boxes[keep],))
                self.assertIsNone(measure(lonely, MAPS))
        self.assertIsNone(measure(Frame(frame.image, ()), MAPS))

    def test_a_text_that_is_not_a_map_is_not_an_anchor(self):
        layout = Layout.default((1920, 1080))
        frame = shown(layout)
        wrong = (replace(frame.boxes[0], text="HELLO"), frame.boxes[1])
        self.assertIsNone(measure(Frame(frame.image, wrong), MAPS))

    def test_a_scale_the_picture_cannot_have_is_rejected(self):
        layout = Layout.default((1920, 1080))
        frame = shown(layout)
        near = TextBox("SKY TEMPLE", 1.0, 0.45, 0.4, 0.1, 0.02)  # right above the second text
        self.assertIsNone(measure(Frame(frame.image, (near, frame.boxes[1])), MAPS))


class TestCanonical(unittest.TestCase):
    def test_a_picture_already_on_the_canvas_is_returned_as_it_is(self):
        for size in (REFERENCE, (1280, 720), (3840, 2160)):
            with self.subTest(size=size):
                layout = Layout.default(size)
                frame = shown(layout)
                self.assertIs(layout.canonical(frame), frame)

    def test_text_and_picture_land_where_the_canvas_has_them(self):
        layout = Layout((1600, 940), 900 / 1080, 800.0, 40.0)
        frame = shown(layout, ("PLAYER", 300.0, 500.0))
        mark_x, mark_y = layout.center + (700 - 960) * layout.scale, layout.top + 300 * layout.scale
        frame.image[int(mark_y) - 4 : int(mark_y) + 4, int(mark_x) - 4 : int(mark_x) + 4] = 255
        canonical = layout.canonical(frame)
        scale = layout.scale
        self.assertEqual(canonical.image.shape, (round(1080 * scale), round(1920 * scale), 3))
        x, y = round(700 * scale), round(300 * scale)
        self.assertGreater(canonical.image[y - 3 : y + 3, x - 3 : x + 3].mean(), 200)
        box = next(b for b in canonical.boxes if b.text == "PLAYER")
        self.assertAlmostEqual(box.cx * REFERENCE[0], 300.0, delta=1.0)
        self.assertAlmostEqual(box.cy * REFERENCE[1], 500.0, delta=1.0)
        self.assertAlmostEqual(box.width * REFERENCE[0], 100.0, delta=1.0)

    def test_a_canvas_wider_than_the_picture_is_padded(self):
        layout = Layout((1280, 720), 0.5, 640.0, 0.0)  # 1920 canvas px are 960 picture px
        canonical = layout.canonical(shown(layout))
        self.assertEqual(canonical.image.shape, (540, 960, 3))


class TestCalibration(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "data" / "calibration.toml"

    def test_an_unmeasured_size_uses_the_default_layout(self):
        frame = shown(Layout.default((1280, 720)))
        frame = Frame(frame.image, ())
        self.assertEqual(Calibration().layout(frame, MAPS), Layout.default((1280, 720)))

    def test_the_measured_layout_is_remembered_for_screens_without_anchors(self):
        layout = Layout((1600, 940), 900 / 1080, 800.0, 40.0)
        calibration = Calibration()
        calibration.layout(shown(layout), MAPS)
        bare = Frame(shown(layout).image, ())
        self.assertTrue(calibration.layout(bare, MAPS).same_as(layout))

    def test_a_moved_window_is_measured_again(self):
        first = Layout((1600, 940), 900 / 1080, 800.0, 40.0)
        moved = Layout((1600, 940), 900 / 1080, 700.0, 10.0)
        calibration = Calibration()
        calibration.layout(shown(first), MAPS)
        self.assertTrue(calibration.layout(shown(moved), MAPS).same_as(moved))
        self.assertTrue(calibration.layout(Frame(shown(moved).image, ()), MAPS).same_as(moved))

    def test_a_tiny_measuring_error_keeps_the_known_layout(self):
        layout = Layout((1600, 940), 900 / 1080, 800.0, 40.0)
        calibration = Calibration()
        known = calibration.layout(shown(layout), MAPS)
        self.assertIs(calibration.layout(shown(layout, noise=1.0), MAPS), known)

    def test_layouts_survive_a_restart(self):
        layout = Layout((1600, 940), 900 / 1080, 800.0, 40.0)
        Calibration(self.path).layout(shown(layout), MAPS)
        bare = Frame(shown(layout).image, ())
        self.assertTrue(Calibration(self.path).layout(bare, MAPS).same_as(layout))

    def test_a_damaged_file_is_ignored(self):
        self.path.parent.mkdir(parents=True)
        for text in ("not [toml", '[layouts."1600x940"]\nscale = 1.0\n'):
            with self.subTest(text=text):
                self.path.write_text(text)
                frame = Frame(shown(Layout.default((1600, 940))).image, ())
                with self.assertLogs("voicemate.hots.screen.layout", "WARNING"):
                    layout = Calibration(self.path).layout(frame, MAPS)
                self.assertEqual(layout, Layout.default((1600, 940)))

    def test_canonical_frames_come_back_on_the_reference_canvas(self):
        layout = Layout((1600, 940), 900 / 1080, 800.0, 40.0)
        frame = shown(layout)
        canonical = Calibration().canonical(frame, MAPS)
        self.assertEqual(canonical.image.shape, (900, 1600, 3))
        self.assertTrue(measure(canonical, MAPS).same_as(Layout.default((1600, 900))))


if __name__ == "__main__":
    unittest.main()
