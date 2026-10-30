"""Spellings Ava learned for hero names, kept between runs.

When a badly pronounced name was settled (Adam picked one of two candidates), the way the
recognizer spelled it is stored here, so the next time it resolves at once.
"""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path

from voicemate.hots.roster import normalize

logger = logging.getLogger(__name__)


class AliasStore:
    """A TOML file of learned spellings (``spelling = "hero-slug"`` under ``[aliases]``).

    Args:
        path: The file; created on the first :meth:`add`.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, str]:
        """The stored ``spelling → slug`` pairs (empty when the file is missing or unreadable)."""
        try:
            with self.path.open("rb") as file:
                return dict(tomllib.load(file).get("aliases", {}))
        except FileNotFoundError:
            return {}
        except (tomllib.TOMLDecodeError, OSError) as exc:
            logger.warning("Ignoring hero aliases in %s: %s", self.path, exc)
            return {}

    def add(self, spelling: str, slug: str) -> None:
        """Remember that ``spelling`` means the hero ``slug`` (written atomically)."""
        aliases = self.load()
        aliases[normalize(spelling)] = slug
        lines = ["[aliases]"] + [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in aliases.items()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(self.path)
