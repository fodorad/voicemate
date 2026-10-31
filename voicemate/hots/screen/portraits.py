"""Reading the ban slots by comparing their pictures with the hero portraits.

A ban shows only the hero's face in one of six slots at the top of the draft screen; the name
appears for two seconds in the middle and is easy to miss. The portraits Adam saved from Icy
Veins show the same art, so each slot is matched against all of them by normalized correlation
(a few scales and shifts), on the CPU with no model. Looks at a slot that did not change are
answered from a cache.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

logger = logging.getLogger(__name__)

#: Slot centres as fractions of the screen width: three bans of Adam's team, three of the enemy's.
SLOT_CENTERS: tuple[float, ...] = (0.1635, 0.215, 0.267, 0.734, 0.786, 0.838)
#: The part of a slot that is compared (fractions of the screen): half-width, top and bottom.
#: It leaves out the hexagon frame and the lock icon below the face.
SLOT_HALF_WIDTH: float = 0.0177
SLOT_TOP: float = 0.0262
SLOT_BOTTOM: float = 0.0707
#: Slots are compared at this size (rows, columns).
PATCH: tuple[int, int] = (24, 34)
#: Portrait widths (in half-resolution pixels) tried against a slot.
SCALES: tuple[int, ...] = (39, 42, 46, 50)
#: Offsets between patch and portrait are tried every this many pixels.
STEP: int = 2
#: A slot counts as a hero when its best correlation reaches this, and beats the runner-up by
#: the margin (empty slots reach about 0.45, true matches 0.65-0.95 on Adam's screens).
MIN_SCORE: float = 0.65
MIN_MARGIN: float = 0.06
#: The file name start of the saved Icy Veins hero portraits.
FILE_PREFIX: str = "img_hero-portraits_"
#: How many slot pictures are remembered (a slot fades in over a few looks).
REMEMBER: int = 256


def _normalized(windows: np.ndarray) -> np.ndarray:
    centred = windows - windows.mean(axis=(-3, -2, -1), keepdims=True)
    norm = np.sqrt((centred * centred).sum(axis=(-3, -2, -1), keepdims=True))
    return centred / (norm + 1e-6)


class Portraits:
    """The hero portraits and the ban-slot matcher.

    Args:
        images: ``slug → portrait`` (any size; RGB).
        remember: How many slot pictures to keep the answer for.
    """

    def __init__(self, images: dict[str, Image.Image], remember: int = REMEMBER) -> None:
        self.slugs = list(images)
        self._scaled = {
            scale: np.stack(
                [
                    np.asarray(
                        image.convert("RGB").resize((scale, scale), Image.Resampling.BILINEAR)
                    )
                    for image in images.values()
                ]
            ).astype(np.float32)
            for scale in SCALES
        }
        self._remember = remember
        self._seen: OrderedDict[bytes, str | None] = OrderedDict()

    @classmethod
    def from_folder(cls, folder: Path) -> Portraits:
        """Load the portraits saved in ``folder`` (``img_hero-portraits_<slug>.jpg``)."""
        images = {}
        for path in sorted(folder.glob(f"{FILE_PREFIX}*")):
            images[path.stem.removeprefix(FILE_PREFIX)] = Image.open(path).convert("RGB")
        logger.info("Loaded %d hero portraits from %s", len(images), folder)
        return cls(images)

    def bans(self, image: np.ndarray) -> tuple[str, ...]:
        """The heroes shown in the six ban slots of a draft screen, left to right."""
        height, width = image.shape[:2]
        found = []
        for center in SLOT_CENTERS:
            box = (
                int((center - SLOT_HALF_WIDTH) * width),
                int(SLOT_TOP * height),
                int((center + SLOT_HALF_WIDTH) * width),
                int(SLOT_BOTTOM * height),
            )
            patch = Image.fromarray(image).crop(box).resize(PATCH[::-1], Image.Resampling.BILINEAR)
            slug = self._identify(np.asarray(patch, dtype=np.float32))
            if slug is not None:
                found.append(slug)
        return tuple(found)

    def _identify(self, patch: np.ndarray) -> str | None:
        key = (patch // 16).astype(np.uint8).tobytes()
        if key in self._seen:
            self._seen.move_to_end(key)
        else:
            self._seen[key] = self._best(patch)
            if len(self._seen) > self._remember:
                self._seen.popitem(last=False)
        return self._seen[key]

    def _best(self, patch: np.ndarray) -> str | None:
        rows, columns = PATCH
        target = _normalized(patch)
        best = np.full(len(self.slugs), -1.0)
        for portraits in self._scaled.values():
            windows = sliding_window_view(portraits, (rows, columns, 3), axis=(1, 2, 3))
            windows = windows[:, ::STEP, ::STEP, 0]
            count = windows.shape[0]
            scores = np.einsum(
                "nkhwc,hwc->nk", _normalized(windows.reshape(count, -1, rows, columns, 3)), target
            )
            best = np.maximum(best, scores.max(axis=1))
        order = np.argsort(best)[::-1]
        if len(order) < 2:
            return self.slugs[order[0]] if len(order) and best[order[0]] >= MIN_SCORE else None
        top, second = best[order[0]], best[order[1]]
        return self.slugs[order[0]] if top >= MIN_SCORE and top - second >= MIN_MARGIN else None
