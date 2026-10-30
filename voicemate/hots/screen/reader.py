"""What a screen read returns, and the seam every screen source implements.

A :class:`ScreenReader` turns "look at the screen" into a :class:`Frame`: the picture (for
colours and portraits) and the text recognized on it. Positions are fractions of the picture
(0-1, measured from the top-left), so nothing downstream depends on the resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: A part of the picture as ``(left, top, right, bottom)`` fractions.
Region = tuple[float, float, float, float]

#: The whole picture.
FULL: Region = (0.0, 0.0, 1.0, 1.0)


class ScreenError(RuntimeError):
    """The screen cannot be read; the message says what to fix."""


@dataclass(frozen=True)
class TextBox:
    """One piece of recognized text.

    Attributes:
        text: The text.
        confidence: Recognizer confidence, 0-1.
        left: Left edge, fraction of the picture width.
        top: Top edge, fraction of the picture height.
        width: Width, fraction of the picture width.
        height: Height, fraction of the picture height.
    """

    text: str
    confidence: float
    left: float
    top: float
    width: float
    height: float

    @property
    def cx(self) -> float:
        """Horizontal centre."""
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        """Vertical centre."""
        return self.top + self.height / 2


@dataclass(frozen=True)
class Frame:
    """One look at the screen.

    Attributes:
        image: The picture, ``(height, width, 3)`` uint8 RGB.
        boxes: Text recognized in the requested region (positions relative to the whole picture).
    """

    image: np.ndarray
    boxes: tuple[TextBox, ...]


class ScreenReader(Protocol):
    """Looks at the screen (or a saved screenshot) and reads the text on it."""

    def read(self, region: Region = FULL, fast: bool = False) -> Frame:
        """Capture the picture and recognize the text inside ``region``.

        Args:
            region: Only text inside this part is recognized (the picture is always whole).
            fast: Use the quicker, rougher recognition (good for big, clear text).

        Raises:
            ScreenError: If the screen cannot be captured.
        """
        ...
