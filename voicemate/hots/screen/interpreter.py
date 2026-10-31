"""What a draft screen says: phase, map, timer and the heroes on each team.

A pure function over a :class:`~voicemate.hots.screen.reader.Frame`: it knows where the game
puts things (hero names in big capitals at the left edge for Adam's team and the right edge for
the enemy, the map name at the top, the phase in the middle) and nothing about how the text was
recognized. Player names are ignored. It also reads the loading screen that follows the draft:
"WELCOME TO <map>" on top and both full teams in rows at the screen's edges.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from voicemate.hots.roster import Roster, normalize, resolve_map
from voicemate.hots.screen.reader import Frame, TextBox

#: Recognized text below this confidence is not trusted for hero names.
MIN_CONFIDENCE: float = 0.5
#: A hero label is locked in when its band is this bright (75th percentile of the label's
#: pixel brightness, 0-255): locked bands are light lavender or white-blue (measured 136-170 on
#: Adam's screens), hovered picks stay dark (at most 97).
LOCKED_BRIGHTNESS: float = 110.0
#: A label is certain (it teaches the hero's face) at this recognizer confidence.
CERTAIN_CONFIDENCE: float = 0.9
#: A timer digit is read at lower confidence (it is one big number).
MIN_TIMER_CONFIDENCE: float = 0.3
#: Hero labels centred left of this belong to Adam's team.
ALLY_EDGE: float = 0.15
#: Hero labels centred right of this belong to the enemy.
ENEMY_EDGE: float = 0.85
#: The middle band of the screen where the phase text sits.
MIDDLE: tuple[float, float] = (0.25, 0.75)
#: The map name sits in the top strip, around the middle.
MAP_STRIP_BOTTOM: float = 0.1
#: The countdown sits just under the map name.
TIMER_BAND: tuple[float, float] = (0.03, 0.15)
#: On the loading screen each hero's name stands in a column at the screen's edge, with the
#: player's name right under it: columns left of / right of these, between these heights.
LOADING_EDGE: float = 0.2
LOADING_ROWS: tuple[float, float] = (0.18, 0.8)
#: Boxes closer than this (fraction of the height) are one row; the hero is its top box.
ROW_GAP: float = 0.04
#: Loading-screen text is read at this confidence (the names sit on busy art).
LOADING_CONFIDENCE: float = 0.3
#: How far above "BANNED" the banned hero's name stands (fraction of the screen height).
BANNED_NAME_GAP: float = 0.08

_BAN = re.compile(r"your ban|red ban|blue ban|ban hero|waiting for enemy ban|bans first|^banning$")
_PICK = re.compile(
    r"blue pick|red pick|lock in|waiting for teammates|waiting for enemy\W*$|^picking$|"
    r"^your team$|^enemy team$"
)
_MINE = re.compile(r"your ban|ban hero|lock in")
_STARTING = re.compile(r"match starting")
_LOADING = re.compile(r"^welcome to$")


@dataclass(frozen=True)
class Phase:
    """Where the draft is.

    Attributes:
        kind: ``"ban"`` or ``"pick"`` during the draft, ``"starting"`` once it is over,
            ``"loading"`` on the loading screen, ``"none"`` for any other screen.
        mine: It is Adam's turn to ban or pick.
    """

    kind: str = "none"
    mine: bool = False


@dataclass(frozen=True)
class DraftScreen:
    """The draft as one screen shows it.

    Attributes:
        phase: The current phase.
        map_slug: The map, when its name was read.
        seconds: The countdown, when it was read.
        ally: Heroes locked in on Adam's team, top slot first.
        enemy: Heroes locked in on the enemy team, top slot first.
        pending: Heroes shown but not locked in yet (hovered picks of either team).
        banned: Heroes named by a "BANNED" announcement.
        invalid: Heroes shown on Adam's side that the enemy has or that are banned.
    """

    phase: Phase = Phase()
    map_slug: str | None = None
    seconds: int | None = None
    ally: tuple[str, ...] = ()
    enemy: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    banned: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()

    @property
    def is_draft(self) -> bool:
        """Whether this is a banning or picking screen."""
        return self.phase.kind in ("ban", "pick")

    @property
    def is_loading(self) -> bool:
        """Whether this is the loading screen (both teams, no draft)."""
        return self.phase.kind == "loading"


class BanReader(Protocol):
    """Recognizes the heroes in the ban slots from the picture."""

    def bans(self, image: np.ndarray) -> tuple[str, ...]:
        """Hero slugs shown in the ban slots of a draft screen."""
        ...


class FaceReader(Protocol):
    """Identifies a locked pick by its portrait when its label's text is in doubt."""

    def check(
        self, image: np.ndarray, side: str, y: float, hero: str | None, certain: bool
    ) -> str | None:
        """The hero of the label at height ``y`` on ``side`` (``hero`` is the text's guess)."""
        ...


def interpret(
    frame: Frame,
    roster: Roster,
    maps: Mapping[str, str],
    portraits: BanReader | None = None,
    faces: FaceReader | None = None,
) -> DraftScreen:
    """Read the draft off one screen.

    Args:
        frame: The captured screen and its text.
        roster: Resolves hero names (the big capital labels).
        maps: ``slug → name`` of the maps.
        portraits: Reads the ban slots; without it only announced bans are known.
        faces: Cross-checks the hero labels against the heroes' portraits (and learns the
            portraits of the labels that are certain); without it the text alone counts.
    """
    boxes = [box for box in frame.boxes if box.text.strip()]
    phase = _phase(boxes)
    if phase.kind == "loading":
        return DraftScreen(
            phase,
            _map(boxes, maps),
            ally=_rows(boxes, roster, "ally"),
            enemy=_rows(boxes, roster, "enemy"),
        )
    ally, ally_pending = _team(frame.image, boxes, roster, "ally", faces)
    enemy, enemy_pending = _team(frame.image, boxes, roster, "enemy", faces)
    banned = _banned(boxes, roster)
    if portraits is not None and phase.kind in ("ban", "pick"):
        banned = tuple(dict.fromkeys((*banned, *portraits.bans(frame.image))))
    taken = {*enemy, *banned}
    shown = [*ally, *ally_pending]
    return DraftScreen(
        phase=phase,
        map_slug=_map(boxes, maps),
        seconds=_seconds(boxes),
        ally=tuple(h for h in ally if h not in taken),
        enemy=enemy,
        pending=tuple(h for h in (*ally_pending, *enemy_pending) if h not in taken),
        banned=banned,
        invalid=tuple(h for h in shown if h in taken),
    )


def _text(box: TextBox) -> str:
    return re.sub(r"\s+", " ", box.text.lower()).strip()


def _phase(boxes: list[TextBox]) -> Phase:
    texts = [_text(b) for b in boxes if MIDDLE[0] < b.cx < MIDDLE[1]]
    top = (_text(b) for b in boxes if b.cy < MAP_STRIP_BOTTOM and MIDDLE[0] < b.cx < MIDDLE[1])
    if any(_LOADING.match(text) for text in top):
        return Phase("loading")
    if any(_STARTING.search(_text(b)) for b in boxes):
        return Phase("starting")
    mine = any(_MINE.search(t) for t in texts)
    if any(_BAN.search(t) for t in texts):
        return Phase("ban", mine)
    if any(_PICK.search(t) for t in texts):
        return Phase("pick", mine)
    return Phase()


def _map(boxes: list[TextBox], maps: Mapping[str, str]) -> str | None:
    for box in boxes:
        if box.cy < MAP_STRIP_BOTTOM and MIDDLE[0] < box.cx < MIDDLE[1]:
            slug = resolve_map(box.text, maps)
            if slug is not None:
                return slug
    return None


def _seconds(boxes: list[TextBox]) -> int | None:
    for box in boxes:
        digits = box.text.strip()
        if (
            digits.isdigit()
            and len(digits) <= 2
            and TIMER_BAND[0] < box.cy < TIMER_BAND[1]
            and abs(box.cx - 0.5) < 0.05
            and box.confidence >= MIN_TIMER_CONFIDENCE
        ):
            return int(digits)
    return None


def _is_label(box: TextBox) -> bool:
    """Hero names are printed in capitals; player names are not."""
    letters = normalize(box.text)
    return box.confidence >= MIN_CONFIDENCE and len(letters) >= 3 and box.text == box.text.upper()


def _locked(image: np.ndarray, box: TextBox) -> bool:
    """Whether the band under a hero label is light (locked in) rather than dark (hovered)."""
    height, width = image.shape[:2]
    top, bottom = int(box.top * height), max(int((box.top + box.height) * height), 1)
    left, right = int(box.left * width), max(int((box.left + box.width) * width), 1)
    patch = image[top:bottom, left:right].astype(np.float32)
    if patch.size == 0:
        return False
    brightness = patch @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(np.percentile(brightness, 75)) >= LOCKED_BRIGHTNESS


def _team(
    image: np.ndarray,
    boxes: list[TextBox],
    roster: Roster,
    side: str,
    faces: FaceReader | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Hero slugs labelled on one side, top slot first: ``(locked, pending)``."""
    locked: list[str] = []
    pending: list[str] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        on_side = box.cx < ALLY_EDGE if side == "ally" else box.cx > ENEMY_EDGE
        if on_side and _is_label(box):
            resolved = roster.resolve(box.text)
            slug = resolved.hero.slug if resolved.hero else None
            is_locked = _locked(image, box)
            if faces is not None and is_locked:
                certain = resolved.exact and box.confidence >= CERTAIN_CONFIDENCE
                slug = faces.check(image, side, box.cy, slug, certain)
            if slug is not None and slug not in locked + pending:
                (locked if is_locked else pending).append(slug)
    return tuple(locked), tuple(pending)


def _rows(boxes: list[TextBox], roster: Roster, side: str) -> tuple[str, ...]:
    """The heroes of one team on the loading screen, top row first."""
    low, high = LOADING_ROWS
    column = sorted(
        (
            b
            for b in boxes
            if (b.cx < LOADING_EDGE if side == "ally" else b.cx > 1 - LOADING_EDGE)
            and low < b.cy < high
            and b.confidence >= LOADING_CONFIDENCE
        ),
        key=lambda b: b.cy,
    )
    tops: list[TextBox] = []
    for box in column:
        if not tops or box.cy - tops[-1].cy >= ROW_GAP:
            tops.append(box)
    heroes: list[str] = []
    for box in tops:
        hero = roster.resolve(box.text).hero
        if hero is not None and hero.slug not in heroes:
            heroes.append(hero.slug)
    return tuple(heroes)


def _banned(boxes: list[TextBox], roster: Roster) -> tuple[str, ...]:
    """Heroes named just above a "BANNED" announcement."""
    heroes: list[str] = []
    for banner in (b for b in boxes if _text(b) == "banned"):
        for box in boxes:
            above = 0 < banner.cy - box.cy < BANNED_NAME_GAP
            if above and abs(box.cx - banner.cx) < 0.1 and _is_label(box):
                hero = roster.resolve(box.text).hero
                if hero is not None and hero.slug not in heroes:
                    heroes.append(hero.slug)
    return tuple(heroes)
