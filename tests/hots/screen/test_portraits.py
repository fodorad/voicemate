import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from voicemate.hots.screen.portraits import SLOT_CENTERS, Portraits

WIDTH, HEIGHT = 960, 540


def portrait(seed: int) -> Image.Image:
    """A blocky random 'face': distinct per seed, smooth enough to survive resizing."""
    blocks = np.random.default_rng(seed).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks).resize((92, 92), Image.Resampling.NEAREST)


def screen(slots: dict[int, Image.Image]) -> np.ndarray:
    """A dark draft screen with the given portraits (by slot number) in the ban slots."""
    image = Image.new("RGB", (WIDTH, HEIGHT), (20, 10, 40))
    for slot, face in slots.items():
        size = 42  # a bit larger than the compared window, as in the game
        face = face.resize((size, size), Image.Resampling.BILINEAR)
        image.paste(face, (int(SLOT_CENTERS[slot] * WIDTH) - size // 2, int(0.054 * HEIGHT) - 21))
    return np.asarray(image)


class TestPortraits(unittest.TestCase):
    def setUp(self):
        self.faces = {
            name: portrait(seed) for seed, name in enumerate(["abathur", "qhira", "zeratul"])
        }
        self.library = Portraits(self.faces)

    def test_filled_slots_are_named_left_to_right(self):
        found = self.library.bans(screen({0: self.faces["qhira"], 3: self.faces["abathur"]}))
        self.assertEqual(found, ("qhira", "abathur"))

    def test_empty_slots_name_nobody(self):
        self.assertEqual(self.library.bans(screen({})), ())

    def test_an_unknown_face_names_nobody(self):
        self.assertEqual(self.library.bans(screen({1: portrait(99)})), ())

    def test_two_identical_portraits_are_too_close_to_tell(self):
        twins = Portraits({"a": self.faces["abathur"], "b": self.faces["abathur"]})
        self.assertEqual(twins.bans(screen({0: self.faces["abathur"]})), ())

    def test_a_second_look_at_the_same_screen_gives_the_same_answer(self):
        image = screen({2: self.faces["zeratul"]})
        self.assertEqual(self.library.bans(image), self.library.bans(image))

    def test_a_tiny_memory_changes_no_answers(self):
        forgetful = Portraits(self.faces, remember=1)
        first, second = screen({0: self.faces["qhira"]}), screen({0: self.faces["abathur"]})
        for image, expected in [(first, "qhira"), (second, "abathur"), (first, "qhira")]:
            self.assertEqual(forgetful.bans(image), (expected,))

    def test_the_picture_size_does_not_matter(self):
        small = np.asarray(Image.fromarray(screen({4: self.faces["qhira"]})).resize((480, 270)))
        self.assertEqual(self.library.bans(small), ("qhira",))

    def test_portraits_load_from_the_saved_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            for slug, face in self.faces.items():
                face.save(Path(tmp) / f"img_hero-portraits_{slug}.png")
            (Path(tmp) / "img_background-images_default.jpg").write_bytes(b"not a portrait")
            loaded = Portraits.from_folder(Path(tmp))
        self.assertEqual(loaded.slugs, ["abathur", "qhira", "zeratul"])
        self.assertEqual(loaded.bans(screen({5: self.faces["abathur"]})), ("abathur",))


if __name__ == "__main__":
    unittest.main()
