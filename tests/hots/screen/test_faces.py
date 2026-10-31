from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.hots.screen.test_interpreter import HEROES, MAPS
from voicemate.hots.screen.faces import (
    CANVAS,
    FACE,
    LABEL_ROW,
    PORTRAIT_X,
    PORTRAIT_Y,
    ROW_STEP,
    Faces,
)
from voicemate.hots.screen.interpreter import interpret
from voicemate.hots.screen.reader import Frame, TextBox

WIDTH, HEIGHT = 960, 540


def texture(seed: int) -> np.ndarray:
    """A hero's face: noise, the same every time for one seed."""
    return np.random.default_rng(seed).integers(0, 256, (80, 100, 3), dtype=np.uint8)


class Screen:
    """A draft screen picture with portraits behind the labels, built slot by slot."""

    def __init__(self) -> None:
        self.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.boxes: list[TextBox] = []

    def pick(
        self, side: str, slot: int, text: str, face: int, confidence: float = 1.0, lit: bool = True
    ) -> Screen:
        x = PORTRAIT_X[side][slot % 2]
        y = PORTRAIT_Y + ROW_STEP * slot
        left, up, right, down = FACE
        x0, y0 = round((x - left) / CANVAS[0] * WIDTH), round((y - up) / CANVAS[1] * HEIGHT)
        x1, y1 = round((x + right) / CANVAS[0] * WIDTH), round((y + down) / CANVAS[1] * HEIGHT)
        patch = texture(face)
        rows = np.linspace(0, patch.shape[0] - 1, y1 - y0).astype(int)
        columns = np.linspace(0, patch.shape[1] - 1, x1 - x0).astype(int)
        self.image[y0:y1, x0:x1] = patch[rows][:, columns]
        cy = (LABEL_ROW + ROW_STEP * slot) / CANVAS[1]
        box = TextBox(text, confidence, 0.01 if side == "ally" else 0.95, cy - 0.02, 0.04, 0.04)
        top, left_px = int(box.top * HEIGHT), int(box.left * WIDTH)
        shade = 200 if lit else 40
        self.image[
            top : top + int(box.height * HEIGHT), left_px : left_px + int(box.width * WIDTH)
        ] = shade
        self.boxes.append(box)
        return self

    def frame(self) -> Frame:
        return Frame(self.image, tuple(self.boxes))


def read(screen: Screen, faces: Faces | None):
    return interpret(screen.frame(), HEROES, MAPS, faces=faces)


class TestFaces(unittest.TestCase):
    def test_a_label_read_with_certainty_teaches_the_face_that_fixes_a_clipped_one(self):
        faces = Faces()
        taught = read(Screen().pick("ally", 2, "JAINA", face=1), faces)
        self.assertEqual(taught.ally, ("jaina",))
        self.assertEqual(len(faces), 1)
        clipped = read(Screen().pick("ally", 0, "QRSTU", face=1), faces)  # another slot, same face
        self.assertEqual(clipped.ally, ("jaina",))
        self.assertEqual(read(Screen().pick("ally", 0, "QRSTU", face=1), None).ally, ())

    def test_an_unknown_face_leaves_the_label_unresolved(self):
        faces = Faces()
        read(Screen().pick("ally", 2, "JAINA", face=1), faces)
        self.assertEqual(read(Screen().pick("ally", 2, "QRSTU", face=2), faces).ally, ())
        self.assertEqual(read(Screen().pick("ally", 2, "QRSTU", face=1), Faces()).ally, ())

    def test_a_doubtful_label_gives_way_to_a_clearly_known_face(self):
        faces = Faces()
        read(Screen().pick("enemy", 1, "THRALL", face=3), faces)
        doubtful = Screen().pick("enemy", 1, "JAINA", face=3, confidence=0.6)
        self.assertEqual(read(doubtful, faces).enemy, ("thrall",))

    def test_a_hovered_pick_is_not_checked(self):
        faces = Faces()
        read(Screen().pick("ally", 2, "JAINA", face=1), faces)
        hovered = Screen().pick("ally", 2, "QRSTU", face=1, lit=False)
        self.assertEqual(read(hovered, faces).pending, ())

    def test_two_heroes_with_the_same_face_are_not_told_apart(self):
        faces = Faces()
        read(Screen().pick("ally", 0, "JAINA", face=1), faces)
        read(Screen().pick("ally", 0, "THRALL", face=1), faces)
        self.assertEqual(read(Screen().pick("ally", 0, "QRSTU", face=1), faces).ally, ())

    def test_a_changed_look_replaces_the_stored_face(self):
        faces = Faces()
        read(Screen().pick("ally", 0, "JAINA", face=1), faces)
        read(Screen().pick("ally", 0, "JAINA", face=2), faces)
        self.assertEqual(len(faces), 1)
        self.assertEqual(read(Screen().pick("ally", 3, "QRSTU", face=2), faces).ally, ("jaina",))

    def test_faces_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "faces"
            read(Screen().pick("ally", 2, "JAINA", face=1), Faces(folder))
            self.assertTrue((folder / "jaina.png").is_file())
            again = Faces(folder)
            self.assertEqual(len(again), 1)
            clipped = read(Screen().pick("ally", 0, "QRSTU", face=1), again)
            self.assertEqual(clipped.ally, ("jaina",))


if __name__ == "__main__":
    unittest.main()
