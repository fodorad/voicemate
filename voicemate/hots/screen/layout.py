"""Where the draft screen sits in the captured picture, measured instead of assumed.

Every position the interpreter and the portrait matcher know is a fraction of the *reference
canvas*: the 1920×1080 draft screen the game draws at full screen. A capture of another size
(a 4K display, a window with a title bar, a laptop's 16:10 panel) shows that canvas scaled and
shifted. :class:`Layout` describes the scale and the shift and maps a :class:`Frame` back onto
the reference canvas (:meth:`Layout.canonical`), so nothing downstream depends on the machine.

The layout is measured from two texts every ban and pick screen shows: the map name at the
top and "Browse All Heroes" below the middle. Their distance gives the scale, the second one
gives the screen's middle line. :class:`Calibration` re-measures whenever those texts are
visible, notices a changed capture (another size, a moved window) and keeps what it measured in
a small file, so a screen without them (the loading screen) is still mapped correctly.
"""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from voicemate.hots.roster import normalize, resolve_map
from voicemate.hots.screen.reader import Frame, TextBox

logger = logging.getLogger(__name__)

#: The size of the reference canvas, in pixels.
REFERENCE: tuple[int, int] = (1920, 1080)
#: The text under the middle of every ban and pick screen, and where its centre stands on the
#: reference canvas (pixels).
BROWSE_TEXT: str = "browseallheroes"
BROWSE_AT: tuple[float, float] = (960.0, 784.4)
#: Where the centre of the map name stands (its x is the middle line, give or take a few pixels).
MAP_AT: tuple[float, float] = (957.2, 25.5)
#: A measurement is trusted when its scale is within this fraction of the one the picture's
#: height suggests (a mismatch means a wrong text was taken for an anchor).
SCALE_TOLERANCE: float = 0.35
#: Two layouts are the same when their scales differ by less than this fraction ...
SAME_SCALE: float = 0.01
#: ... and their offsets by less than this many reference pixels.
SAME_OFFSET: float = 4.0
#: Anchor text is trusted at this recognizer confidence.
MIN_CONFIDENCE: float = 0.5


@dataclass(frozen=True)
class Layout:
    """How the reference canvas appears in a picture of one size.

    Attributes:
        size: ``(width, height)`` of the captured picture, pixels.
        scale: Picture pixels per reference pixel.
        center: Picture x of the canvas' middle line, pixels.
        top: Picture y of the canvas' top edge, pixels (negative when it is cut off).
    """

    size: tuple[int, int]
    scale: float
    center: float
    top: float

    @classmethod
    def default(cls, size: tuple[int, int]) -> Layout:
        """The canvas filling the picture's height and centred in its width."""
        width, height = size
        return cls(size, height / REFERENCE[1], width / 2, 0.0)

    def same_as(self, other: Layout) -> bool:
        """Whether ``other`` places the canvas in (nearly) the same way."""
        scale = self.scale
        return (
            self.size == other.size
            and abs(scale - other.scale) <= SAME_SCALE * scale
            and abs(self.center - other.center) <= SAME_OFFSET * scale
            and abs(self.top - other.top) <= SAME_OFFSET * scale
        )

    @property
    def is_identity(self) -> bool:
        """Whether the picture already is the canvas (at any resolution)."""
        width, height = self.size
        tolerance = SAME_OFFSET * self.scale
        return (
            abs(width - REFERENCE[0] * self.scale) <= tolerance
            and abs(height - REFERENCE[1] * self.scale) <= tolerance
            and abs(self.center - width / 2) <= tolerance
            and abs(self.top) <= tolerance
        )

    def canonical(self, frame: Frame) -> Frame:
        """``frame`` cut to the canvas: the same pixels, so only its size and text positions change.

        Anything the picture does not show of the canvas is black.
        """
        if self.is_identity:
            return frame
        width, height = self.size
        scale = self.scale
        left = self.center - REFERENCE[0] / 2 * scale
        box = (
            round(left),
            round(self.top),
            round(left + REFERENCE[0] * scale),
            round(self.top + REFERENCE[1] * scale),
        )
        image = Image.fromarray(frame.image).crop(box)
        boxes = tuple(
            TextBox(
                b.text,
                b.confidence,
                (b.left * width - left) / scale / REFERENCE[0],
                (b.top * height - self.top) / scale / REFERENCE[1],
                b.width * width / scale / REFERENCE[0],
                b.height * height / scale / REFERENCE[1],
            )
            for b in frame.boxes
        )
        return Frame(np.asarray(image), boxes)


def measure(frame: Frame, maps: Mapping[str, str]) -> Layout | None:
    """The layout a draft screen shows, or ``None`` when its anchor texts are not both readable."""
    height, width = frame.image.shape[:2]
    browse = [
        b
        for b in frame.boxes
        if b.confidence >= MIN_CONFIDENCE and normalize(b.text) == BROWSE_TEXT
    ]
    if not browse:
        return None
    below = browse[0]
    names = sorted(
        (
            b
            for b in frame.boxes
            if b.cy < below.cy
            and abs(b.cx - below.cx) < 0.05
            and b.confidence >= MIN_CONFIDENCE
            and resolve_map(b.text, maps) is not None
        ),
        key=lambda b: b.cy,
    )
    if not names:
        return None
    top_y, browse_y = names[0].cy * height, below.cy * height
    scale = (browse_y - top_y) / (BROWSE_AT[1] - MAP_AT[1])
    expected = height / REFERENCE[1]
    if abs(scale / expected - 1) > SCALE_TOLERANCE:
        return None
    return Layout((width, height), scale, below.cx * width, top_y - MAP_AT[1] * scale)


class Calibration:
    """Finds the layout of each frame and remembers what it measured.

    Args:
        path: TOML file where measured layouts are kept between runs (memory only when omitted).
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._known: dict[tuple[int, int], Layout] = self._load()

    def layout(self, frame: Frame, maps: Mapping[str, str]) -> Layout:
        """The layout of ``frame``: measured when it shows its anchors, otherwise remembered.

        A measurement that differs from the remembered layout (another capture size, a moved
        window, a changed game setting) replaces it and is stored.
        """
        height, width = frame.image.shape[:2]
        size = (width, height)
        measured = measure(frame, maps)
        known = self._known.get(size)
        if measured is None:
            return known or Layout.default(size)
        if known is None or not measured.same_as(known):
            logger.info("Screen layout %dx%d measured: %s", width, height, measured)
            self._known[size] = measured
            self._save()
            return measured
        return known

    def canonical(self, frame: Frame, maps: Mapping[str, str]) -> Frame:
        """``frame`` on the reference canvas, whatever the machine's screen looks like."""
        return self.layout(frame, maps).canonical(frame)

    def _load(self) -> dict[tuple[int, int], Layout]:
        if self.path is None:
            return {}
        try:
            with self.path.open("rb") as file:
                stored = tomllib.load(file).get("layouts", {})
            known = {}
            for key, value in stored.items():
                width, height = (int(part) for part in key.split("x"))
                known[(width, height)] = Layout(
                    (width, height),
                    float(value["scale"]),
                    float(value["center"]),
                    float(value["top"]),
                )
            return known
        except FileNotFoundError:
            return {}
        except (tomllib.TOMLDecodeError, OSError, KeyError, ValueError) as exc:
            logger.warning("Ignoring screen calibration in %s: %s", self.path, exc)
            return {}

    def _save(self) -> None:
        if self.path is None:
            return
        lines: list[str] = []
        for (width, height), layout in self._known.items():
            lines.append(f"[layouts.{json.dumps(f'{width}x{height}')}]")
            lines.append(f"scale = {layout.scale:.5f}")
            lines.append(f"center = {layout.center:.2f}")
            lines.append(f"top = {layout.top:.2f}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(self.path)
