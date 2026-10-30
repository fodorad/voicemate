"""What a draft screen says: phase, map, timer and the heroes on each team.

A pure function over a :class:`~voicemate.hots.screen.reader.Frame`: it knows where the game
puts things (hero names in big capitals at the left edge for Adam's team and the right edge for
the enemy, the map name at the top, the phase in the middle) and nothing about how the text was
recognized. Player names are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from voicemate.hots.roster import Roster, normalize, resolve_map
from voicemate.hots.screen.reader import Frame, TextBox

#: Recognized text below this confidence is not trusted for hero names.
MIN_CONFIDENCE: float = 0.5
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
#: How far above "BANNED" the banned hero's name stands (fraction of the screen height).
BANNED_NAME_GAP: float = 0.08

_BAN = re.compile(r"your ban|red ban|blue ban|ban hero|waiting for enemy ban|bans first|^banning$")
_PICK = re.compile(
    r"blue pick|red pick|lock in|waiting for teammates|waiting for enemy\W*$|^picking$|"
    r"^your team$|^enemy team$"
)
_MINE = re.compile(r"your ban|ban hero|lock in")
_STARTING = re.compile(r"match starting")


@dataclass(frozen=True)
class Phase:
    """Where the draft is.

    Attributes:
        kind: ``"ban"`` or ``"pick"`` during the draft, ``"starting"`` once it is over,
            ``"none"`` when this is not a draft screen.
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
        ally: Heroes on Adam's team, top slot first.
        enemy: Heroes on the enemy team, top slot first.
        banned: Heroes named by a "BANNED" announcement.
        invalid: Heroes shown on Adam's side that the enemy has or that are banned.
    """

    phase: Phase = Phase()
    map_slug: str | None = None
    seconds: int | None = None
    ally: tuple[str, ...] = ()
    enemy: tuple[str, ...] = ()
    banned: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()

    @property
    def is_draft(self) -> bool:
        """Whether this is a banning or picking screen."""
        return self.phase.kind in ("ban", "pick")


def interpret(frame: Frame, roster: Roster, maps: Mapping[str, str]) -> DraftScreen:
    """Read the draft off one screen.

    Args:
        frame: The captured screen and its text.
        roster: Resolves hero names (the big capital labels).
        maps: ``slug → name`` of the maps.
    """
    boxes = [box for box in frame.boxes if box.text.strip()]
    ally, enemy = (_team(boxes, roster, side) for side in ("ally", "enemy"))
    banned = _banned(boxes, roster)
    taken = {*enemy, *banned}
    return DraftScreen(
        phase=_phase(boxes),
        map_slug=_map(boxes, maps),
        seconds=_seconds(boxes),
        ally=tuple(h for h in ally if h not in taken),
        enemy=enemy,
        banned=banned,
        invalid=tuple(h for h in ally if h in taken),
    )


def _text(box: TextBox) -> str:
    return re.sub(r"\s+", " ", box.text.lower()).strip()


def _phase(boxes: list[TextBox]) -> Phase:
    texts = [_text(b) for b in boxes if MIDDLE[0] < b.cx < MIDDLE[1]]
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


def _team(boxes: list[TextBox], roster: Roster, side: str) -> tuple[str, ...]:
    """Hero slugs labelled on one side, top slot first."""
    heroes: list[str] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        on_side = box.cx < ALLY_EDGE if side == "ally" else box.cx > ENEMY_EDGE
        if on_side and _is_label(box):
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
