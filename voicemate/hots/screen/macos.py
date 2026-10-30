"""The macOS screen readers: Quartz capture and Apple Vision text recognition.

Both run on the CPU/Neural Engine with no model of ours: a full read takes about half a
second. Capturing another app's window needs the Screen Recording permission for the app
that launched Ava (System Settings → Privacy & Security → Screen & System Audio Recording).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from voicemate.hots.screen.reader import FULL, Frame, Region, ScreenError, TextBox

logger = logging.getLogger(__name__)

PERMISSION_HINT = (
    "Allow Screen Recording for the app that runs Ava (Terminal, iTerm or VS Code) in "
    "System Settings > Privacy & Security > Screen & System Audio Recording, then restart it."
)


def recognize(image: np.ndarray, region: Region = FULL, fast: bool = False) -> tuple[TextBox, ...]:
    """Recognize the text inside ``region`` of an RGB picture with Apple Vision."""
    from ocrmac import ocrmac
    from PIL import Image

    height, width = image.shape[:2]
    left, top, right, bottom = region
    crop = image[int(top * height) : int(bottom * height), int(left * width) : int(right * width)]
    level = "fast" if fast else "accurate"
    found = ocrmac.OCR(
        Image.fromarray(crop), recognition_level=level, language_preference=["en-US"]
    ).recognize()
    boxes = []
    for text, confidence, (x, y, w, h) in found:
        # Vision measures y upward from the bottom of the crop; fractions of the crop → picture.
        box_top = top + (1.0 - y - h) * (bottom - top)
        boxes.append(
            TextBox(
                text,
                float(confidence),
                left + x * (right - left),
                box_top,
                w * (right - left),
                h * (bottom - top),
            )
        )
    return tuple(boxes)


def _to_array(image) -> np.ndarray:
    """A Quartz ``CGImage`` as an RGB array."""
    import Quartz

    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    per_row = Quartz.CGImageGetBytesPerRow(image)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, per_row // 4, 4)
    return np.ascontiguousarray(pixels[:, :width, [2, 1, 0]])  # BGRA → RGB


def _window_id(name: str) -> int | None:
    """The biggest on-screen window whose app or title contains ``name``."""
    import Quartz

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    wanted = name.lower()
    best, area = None, 0.0
    for window in windows:
        label = f"{window.get('kCGWindowOwnerName', '')} {window.get('kCGWindowName', '')}"
        bounds = window.get("kCGWindowBounds", {})
        size = bounds.get("Width", 0) * bounds.get("Height", 0)
        if wanted in label.lower() and size > area:
            best, area = int(window["kCGWindowNumber"]), size
    return best


def capture(window: str) -> np.ndarray:
    """The picture of the ``window`` app's window, or of the whole screen when it is not open.

    Raises:
        ScreenError: If macOS returns nothing or a blank picture (permission missing).
    """
    import Quartz

    window_id = _window_id(window) if window else None
    if window_id is None:
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectInfinite,
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
            Quartz.kCGWindowImageDefault,
        )
    else:
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )
    if image is None:
        raise ScreenError(f"The screen could not be captured. {PERMISSION_HINT}")
    pixels = _to_array(image)
    if pixels.size == 0 or int(pixels.max()) == int(pixels.min()):
        raise ScreenError(f"The screen capture is blank. {PERMISSION_HINT}")
    return pixels


class MacScreenReader:
    """Reads the live screen (the game window when it is open, otherwise the whole display).

    Args:
        window: Part of the game's app or window name (``"Heroes"``).
    """

    def __init__(self, window: str = "Heroes") -> None:
        self.window = window

    def read(self, region: Region = FULL, fast: bool = False) -> Frame:
        """Capture the screen now and recognize the text inside ``region``."""
        image = capture(self.window)
        return Frame(image, recognize(image, region, fast))


class ImageFileReader:
    """Reads a saved screenshot instead of the live screen.

    Args:
        newest: Returns the file to read (e.g. the newest screenshot in a folder).
    """

    def __init__(self, newest: Callable[[], Path | None]) -> None:
        self.newest = newest

    def read(self, region: Region = FULL, fast: bool = False) -> Frame:
        """Load the screenshot and recognize the text inside ``region``.

        Raises:
            ScreenError: If there is no screenshot.
        """
        from PIL import Image

        path = self.newest()
        if path is None:
            raise ScreenError("There is no saved screenshot to read.")
        image = np.asarray(Image.open(path).convert("RGB"))
        return Frame(image, recognize(image, region, fast))
