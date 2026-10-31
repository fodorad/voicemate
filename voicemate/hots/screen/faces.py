"""Telling heroes apart by their pick portraits, learned from labels that were read with certainty.

The big hexagon portraits of a pick are another painting than the small Icy Veins portraits
(measured: no correlation), so they cannot be looked up. What the game draws for a hero is the
same every time, though, so each label read without doubt (an exact name, clear text, a locked
pick) teaches this gallery the hero's face. A label that is later misread (a clipped letter on
rotated text, a low resolution) is then identified by its face instead. The gallery is kept in
a folder, one small picture per hero, so it fills up over a few drafts and survives restarts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: Label centre of the top slot and the distance between slots (reference pixels, 1080 high).
LABEL_ROW: float = 184.0
ROW_STEP: float = 170.0
#: Portrait centre of the top slot; slots zig-zag, so the x of even and odd slots differs
#: (reference pixels, 1920 wide).
PORTRAIT_Y: float = 150.0
PORTRAIT_X: dict[str, tuple[float, float]] = {"ally": (115.0, 215.0), "enemy": (1805.0, 1705.0)}
#: The part of a portrait that is compared, as distances from its centre (reference pixels):
#: left, up, right, down. It leaves out the name band across the lower part.
FACE: tuple[float, float, float, float] = (50.0, 70.0, 50.0, 10.0)
#: Faces are stored and compared at this size (width, height).
SIZE: tuple[int, int] = (50, 40)
#: A face is the same hero at this correlation (measured: the same hero reaches 0.88 and more,
#: different heroes stay under 0.5) and must beat every other face by the margin.
MIN_SCORE: float = 0.8
MIN_MARGIN: float = 0.2
#: Reference canvas size the slot positions are given in.
CANVAS: tuple[int, int] = (1920, 1080)
FILE_SUFFIX: str = ".png"


def _normalized(face: np.ndarray) -> np.ndarray:
    centred = face.astype(np.float32).ravel()
    centred -= centred.mean()
    return centred / (float(np.sqrt((centred * centred).sum())) + 1e-6)


class Faces:
    """The gallery of hero faces.

    Args:
        folder: Where the faces are kept between runs (memory only when omitted).
    """

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._faces: dict[str, np.ndarray] = {}
        self._stack: np.ndarray | None = None
        if folder is not None and folder.is_dir():
            for path in sorted(folder.glob(f"*{FILE_SUFFIX}")):
                self._faces[path.stem] = np.asarray(Image.open(path).convert("RGB"))
            logger.info("Loaded %d hero faces from %s", len(self._faces), folder)

    def __len__(self) -> int:
        """Number of heroes whose face is known."""
        return len(self._faces)

    def check(
        self, image: np.ndarray, side: str, y: float, hero: str | None, certain: bool
    ) -> str | None:
        """The hero a locked pick label stands for, by its face when the text is in doubt.

        A label read with certainty teaches its face and is returned as it is; any other label
        is identified by the face when the gallery knows it clearly, otherwise the text's
        guess (``hero``, possibly ``None``) stays.

        Args:
            image: The draft screen on the reference canvas (any resolution).
            side: ``"ally"`` or ``"enemy"``.
            y: The label's vertical centre, fraction of the picture height.
            hero: The hero the label's text resolved to, if any.
            certain: Whether the text can be trusted without the face.
        """
        face = self._crop(image, side, y)
        if certain and hero is not None:
            self._learn(hero, face)
            return hero
        return self._identify(face) or hero

    def _crop(self, image: np.ndarray, side: str, y: float) -> np.ndarray:
        height, width = image.shape[:2]
        slot = min(max(round((y * CANVAS[1] - LABEL_ROW) / ROW_STEP), 0), 4)
        x = PORTRAIT_X[side][slot % 2]
        center_y = PORTRAIT_Y + ROW_STEP * slot
        left, up, right, down = FACE
        box = (
            round((x - left) / CANVAS[0] * width),
            round((center_y - up) / CANVAS[1] * height),
            round((x + right) / CANVAS[0] * width),
            round((center_y + down) / CANVAS[1] * height),
        )
        return np.asarray(Image.fromarray(image).crop(box).resize(SIZE, Image.Resampling.BILINEAR))

    def _identify(self, face: np.ndarray) -> str | None:
        if not self._faces:
            return None
        slugs = list(self._faces)
        if self._stack is None:
            self._stack = np.stack([_normalized(self._faces[slug]) for slug in slugs])
        scores = self._stack @ _normalized(face)
        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0
        if best >= MIN_SCORE and best - second >= MIN_MARGIN:
            return slugs[int(order[0])]
        return None

    def _learn(self, hero: str, face: np.ndarray) -> None:
        known = self._faces.get(hero)
        if known is not None and float(_normalized(known) @ _normalized(face)) >= MIN_SCORE:
            return
        self._faces[hero] = face
        self._stack = None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(face).save(self.folder / f"{hero}{FILE_SUFFIX}")
