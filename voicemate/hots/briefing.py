"""The team briefing: what the two full teams mean for the match, in a few short sentences.

Where the :class:`~voicemate.hots.advisor.Advisor` ranks heroes to pick, the briefing reads a
finished lineup (the loading screen shows both teams) and says what to do with it. It weighs
four kinds of facts, each worth a note with a weight, and says the heaviest few:

* **Shape of the teams:** no healer, no frontline, a squishy lineup, and the same for the enemy.
* **Matchups:** which of your heroes counter an enemy hero (focus it) and which enemy heroes
  counter one of yours (protect it), from the counter lists of the hero pages.
* **The map:** heroes that are stronger or weaker on this map than elsewhere.
* **Combos:** two of your heroes that the pages list as working well together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from voicemate.hots.advisor import Advisor
from voicemate.hots.roster import Hero

#: How many notes a briefing says.
NOTES: int = 3
#: How heavy each kind of note is (higher is said first).
NO_HEALER: float = 9.0
NO_FRONTLINE: float = 8.0
COUNTERED: float = 7.0
COUNTERS: float = 6.0
SQUISHY: float = 6.0
ENEMY_SHAPE: float = 5.0
MAP_NOTE: float = 4.0
COMBO: float = 3.0
#: A team with this many assassins and at most one frontline hero is squishy.
SQUISHY_ASSASSINS: int = 4
#: Names listed in one sentence.
NAMES: int = 2
#: Said when nothing stands out.
EVEN: str = "No clear edge either way."
#: Said when there is nothing to brief.
NO_TEAMS: str = "I know no teams yet."


@dataclass(frozen=True)
class Note:
    """One thing worth saying.

    Attributes:
        weight: How important it is (higher is said first).
        text: The short sentence(s).
    """

    weight: float
    text: str


class _Team:
    """A team's heroes counted by role."""

    def __init__(self, heroes: Sequence[Hero]) -> None:
        roles = [hero.role.lower() for hero in heroes]
        self.healers = sum("healer" in role for role in roles)
        self.supports = sum("support" in role for role in roles)
        self.frontline = sum("tank" in role or "bruiser" in role for role in roles)
        self.assassins = sum("assassin" in role for role in roles)

    @property
    def squishy(self) -> bool:
        return self.assassins >= SQUISHY_ASSASSINS and self.frontline <= 1


class Briefing:
    """Says what a lineup means.

    Args:
        advisor: Knows the heroes, their counters, synergies and map ratings.
    """

    def __init__(self, advisor: Advisor) -> None:
        self.advisor = advisor

    def say(self, ally: Sequence[str], enemy: Sequence[str], map_slug: str | None = None) -> str:
        """The briefing for a lineup: the most important notes, as short sentences.

        Args:
            ally: Slugs of your team.
            enemy: Slugs of the enemy team.
            map_slug: The map, when known.
        """
        if not self._heroes(ally) and not self._heroes(enemy):
            return NO_TEAMS
        notes = sorted(self.notes(ally, enemy, map_slug), key=lambda note: -note.weight)
        return " ".join(note.text for note in notes[:NOTES]) or EVEN

    def notes(self, ally: Sequence[str], enemy: Sequence[str], map_slug: str | None) -> list[Note]:
        """Every note for a lineup, unsorted."""
        mine, theirs = self._heroes(ally), self._heroes(enemy)
        notes = [*self._shape(_Team(mine), _Team(theirs), bool(mine), bool(theirs))]
        notes += self._matchups(mine, theirs)
        notes += self._map(mine, theirs, map_slug)
        notes += self._combos(mine)
        return notes

    # ── kinds of notes ───────────────────────────────────────────────────────

    @staticmethod
    def _shape(mine: _Team, theirs: _Team, have_mine: bool, have_theirs: bool) -> list[Note]:
        notes = []
        if have_mine:
            if mine.healers == 0 and mine.supports == 0:
                notes.append(Note(NO_HEALER, "No healer: don't overextend."))
            elif mine.healers == 0:
                notes.append(Note(NO_HEALER - 4, "No healer: play safe."))
            if mine.frontline == 0:
                notes.append(Note(NO_FRONTLINE, "No frontline: don't fight in the open."))
            if mine.squishy:
                notes.append(Note(SQUISHY, "Squishy team: stay grouped."))
        if have_theirs:
            if theirs.healers == 0:
                notes.append(Note(ENEMY_SHAPE, "No enemy healer: keep trading."))
            elif theirs.healers >= 2:
                notes.append(Note(ENEMY_SHAPE, "Two enemy healers: kill one first."))
            if theirs.frontline == 0 or theirs.squishy:
                notes.append(Note(ENEMY_SHAPE, "Enemy is squishy: dive them."))
        return notes

    def _matchups(self, mine: list[Hero], theirs: list[Hero]) -> list[Note]:
        knowledge = self.advisor.knowledge
        notes = []
        for friend in mine:
            beaten = [h for h in theirs if friend.slug in knowledge.counters_of(h.slug)]
            if beaten:
                text = f"{friend.name} counters {_names(beaten)}."
                notes.append(Note(COUNTERS + len(beaten), text))
        for foe in theirs:
            hunted = [h for h in mine if foe.slug in knowledge.counters_of(h.slug)]
            if hunted:
                text = f"{foe.name} counters {_names(hunted)}: protect."
                notes.append(Note(COUNTERED + len(hunted), text))
        return notes

    def _map(self, mine: list[Hero], theirs: list[Hero], map_slug: str | None) -> list[Note]:
        if not map_slug:
            return []
        where = self.advisor.maps.get(map_slug, map_slug)
        rate = self.advisor.knowledge.map_rating
        notes = []
        for label, heroes, rating in (
            (f"Strong on {where}", mine, "stronger"),
            (f"Weak on {where}", mine, "weaker"),
            (f"Enemy strong on {where}", theirs, "stronger"),
        ):
            picked = [h for h in heroes if rate(h.slug, map_slug) == rating]
            if picked:
                notes.append(Note(MAP_NOTE, f"{label}: {_names(picked)}."))
        return notes

    def _combos(self, mine: list[Hero]) -> list[Note]:
        knowledge = self.advisor.knowledge
        for index, first in enumerate(mine):
            for second in mine[index + 1 :]:
                liked = set(knowledge.synergies_of(first.slug))
                if second.slug in liked or first.slug in knowledge.synergies_of(second.slug):
                    return [Note(COMBO, f"{first.name} and {second.name} combo well.")]
        return []

    def _heroes(self, slugs: Sequence[str]) -> list[Hero]:
        return [hero for slug in slugs if (hero := self.advisor.roster.get(slug))]


def _names(heroes: Sequence[Hero]) -> str:
    """The first names, joined: ``A`` or ``A and B``."""
    return " and ".join(hero.name for hero in heroes[:NAMES])
