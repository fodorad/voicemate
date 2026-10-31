"""Spoken commands about the screen and draft mode (everything else is a draft question)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LAST = re.compile(
    r"\b(?:last|latest|previous|saved) (?:screenshot|screen|image|picture)\b|"
    r"\b(?:read|scan|check) (?:the )?screenshot\b"
)
_READ = re.compile(
    r"^(?:(?:read|scan|check|look at|watch)(?: the| my| this)? (?:draft|screen|picks|bans|game)\b|"
    r"(?:read|scan)$|what do you see\b)"
)
_ON = re.compile(r"^(?:(?:start|enter|begin|turn on|switch to) )?draft mode(?: on)?$|^start draft$")
_OFF = re.compile(
    r"^(?:(?:end|stop|leave|exit|finish|cancel|turn off) (?:the )?draft(?: mode)?|"
    r"draft mode off|(?:the )?draft is over)$"
)


@dataclass(frozen=True)
class Command:
    """A recognized screen or mode command.

    Attributes:
        kind: ``"read"`` (look at the screen), ``"last"`` (read the newest saved screenshot),
            ``"on"`` or ``"off"`` (draft mode).
        advice: What to advise after reading: ``"ban"``, ``"pick"`` or ``""`` (what fits the
            phase).
    """

    kind: str
    advice: str = ""


def parse(said: str) -> Command | None:
    """The command in ``said`` (already cleaned, see ``intents.clean``), or ``None``."""
    advice = "ban" if re.search(r"\bban\b", said) else "pick" if "pick" in said else ""
    if _OFF.match(said):
        return Command("off")
    if _ON.match(said):
        return Command("on")
    if _LAST.search(said):
        return Command("last", advice)
    if _READ.match(said):
        return Command("read", advice)
    return None
