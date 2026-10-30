"""The log of captured draft screens: one folder per draft, numbered screenshots inside.

``data/game_screens/<YYYY-MM-DD_HHMMSS>/01.png, 02.png, … + draft.json`` lets Adam check later
which screens Ava saw and what she made of each. Old drafts are deleted so the folder stays
small.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: Name of a draft folder (anything else in the log folder is never touched).
_DRAFT_FOLDER = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:-\d+)?$")

#: The per-draft summary file.
SUMMARY_FILE = "draft.json"


class ScreenLog:
    """Saves the screens of drafts.

    Args:
        folder: The log folder (created when the first draft starts).
        keep: How many draft folders to keep; the oldest are deleted when a draft starts.
        clock: Current time (replaceable for tests).
    """

    def __init__(
        self, folder: Path, keep: int = 20, clock: Callable[[], datetime] = datetime.now
    ) -> None:
        self.folder = folder
        self.keep = keep
        self._clock = clock
        self._current: Path | None = None

    def start_draft(self) -> Path:
        """Open the folder of a new draft and delete the oldest drafts beyond ``keep``."""
        now = self._clock()
        name = now.strftime("%Y-%m-%d_%H%M%S")
        path, number = self.folder / name, 1
        while path.exists():
            number += 1
            path = self.folder / f"{name}-{number}"
        path.mkdir(parents=True)
        self._write(path, {"started": now.isoformat(timespec="seconds"), "screens": []})
        self._current = path
        for old in self._drafts()[: -self.keep]:
            logger.info("Removing old draft screens %s", old.name)
            shutil.rmtree(old)
        return path

    def save(self, image: np.ndarray, state: dict[str, Any]) -> Path:
        """Save ``image`` as the next screen of the current draft with what was read from it.

        Starts a draft first when none is open.

        Args:
            image: The captured picture, RGB uint8.
            state: What Ava made of it (JSON-able); stored in ``draft.json``.

        Returns:
            The saved PNG.
        """
        draft = self._current or self.start_draft()
        summary = json.loads((draft / SUMMARY_FILE).read_text(encoding="utf-8"))
        name = f"{len(summary['screens']) + 1:02d}.png"
        Image.fromarray(image).save(draft / name)
        summary["screens"].append(
            {"file": name, "at": self._clock().isoformat(timespec="seconds"), **state}
        )
        self._write(draft, summary)
        return draft / name

    def newest_screenshot(self) -> Path | None:
        """The latest saved screen of the newest draft, or ``None`` when nothing was saved."""
        for draft in reversed(self._drafts()):
            screens = sorted(draft.glob("*.png"))
            if screens:
                return screens[-1]
        return None

    def _drafts(self) -> list[Path]:
        """Draft folders, oldest first."""
        if not self.folder.is_dir():
            return []
        return sorted(
            p for p in self.folder.iterdir() if p.is_dir() and _DRAFT_FOLDER.match(p.name)
        )

    @staticmethod
    def _write(draft: Path, summary: dict[str, Any]) -> None:
        (draft / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
