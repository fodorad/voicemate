"""Draft advice: counters, picks and bans ranked from the knowledge base.

Every ranking is a sum of a few named weights (below) so Adam can tune them, and every
suggestion carries the reasons that fed its score so an answer can say *why*. Missing data
only removes factors: a hero without counters or map ratings is still scored by what is known.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from voicemate.hots.knowledge import Knowledge
from voicemate.hots.roster import Hero

#: Score for each enemy hero the candidate counters (picks and counters).
COUNTER_HIT: float = 2.0
#: Score taken off for each enemy hero that counters the candidate.
COUNTERED_BY: float = 1.0
#: Score for each ally the candidate works well with (picks).
ALLY_SYNERGY: float = 0.5
#: Score for each enemy hero a banned candidate works well with (bans).
ENEMY_SYNERGY: float = 1.0
#: Score for each ally a banned candidate would counter (bans).
ALLY_COUNTERED: float = 1.0
#: Score when the tier list recommends banning the hero (bans).
BAN_FLAG: float = 1.0
#: Score for how well the hero does on the map.
MAP_RATING: dict[str, float] = {"stronger": 1.0, "average": 0.0, "weaker": -1.0}
#: Score for the hero's tier on the map's (or else the general) tier list.
TIER_BONUS: dict[str, float] = {"S": 1.0, "A": 0.5, "B": 0.0, "C": -0.5, "D": -1.0}


@dataclass(frozen=True)
class Suggestion:
    """One ranked hero.

    Attributes:
        hero: The suggested hero.
        score: Sum of the weights that applied (higher is better).
        reasons: Short facts behind the score, strongest first.
    """

    hero: Hero
    score: float
    reasons: tuple[str, ...]


class Advisor:
    """Ranks heroes for the draft.

    Args:
        knowledge: The imported pages.
    """

    def __init__(self, knowledge: Knowledge) -> None:
        self.knowledge = knowledge
        self.roster = knowledge.roster()
        self.maps = knowledge.maps()

    def counters(self, hero: str, map_slug: str | None = None, limit: int = 3) -> list[Suggestion]:
        """Heroes that counter ``hero``, best first.

        Args:
            hero: Slug of the hero to counter.
            map_slug: Map being played, if known.
            limit: Number of suggestions.
        """
        candidates = [self.roster.get(slug) for slug in self.knowledge.counters_of(hero)]
        ranked = []
        for candidate in filter(None, candidates):
            score, reasons = COUNTER_HIT, [f"counters {self._name(hero)}"]
            extra, why = self._setting(candidate.slug, map_slug)
            ranked.append(Suggestion(candidate, score + extra, tuple(reasons + why)))
        return self._top(ranked, limit)

    def picks(
        self,
        enemy: Collection[str],
        ally: Collection[str] = (),
        map_slug: str | None = None,
        banned: Collection[str] = (),
        limit: int = 3,
    ) -> list[Suggestion]:
        """Heroes to pick against the enemy picks, with the allies and the map in mind."""
        taken = {*enemy, *ally, *banned}
        enemy_countered = {e: set(self.knowledge.counters_of(e)) for e in enemy}
        ally_synergy = {a: set(self.knowledge.synergies_of(a)) for a in ally}
        ranked = []
        for candidate in self.roster:
            if candidate.slug in taken:
                continue
            score, reasons = 0.0, []
            own_counters = set(self.knowledge.counters_of(candidate.slug))
            own_synergies = set(self.knowledge.synergies_of(candidate.slug))
            for e in enemy:
                if candidate.slug in enemy_countered[e]:
                    score += COUNTER_HIT
                    reasons.append(f"counters {self._name(e)}")
                if e in own_counters:
                    score -= COUNTERED_BY
            for a in ally:
                if candidate.slug in ally_synergy[a] or a in own_synergies:
                    score += ALLY_SYNERGY
                    reasons.append(f"works with {self._name(a)}")
            extra, why = self._setting(candidate.slug, map_slug)
            ranked.append(Suggestion(candidate, score + extra, tuple(reasons + why)))
        return self._top(ranked, limit)

    def bans(
        self,
        enemy: Collection[str],
        ally: Collection[str] = (),
        map_slug: str | None = None,
        banned: Collection[str] = (),
        limit: int = 3,
    ) -> list[Suggestion]:
        """Heroes to ban: strong on the map, synergizing with the enemy picks so far."""
        taken = {*enemy, *ally, *banned}
        enemy_synergy = {e: set(self.knowledge.synergies_of(e)) for e in enemy}
        ranked = []
        for candidate in self.roster:
            if candidate.slug in taken:
                continue
            score, reasons = 0.0, []
            own_synergies = set(self.knowledge.synergies_of(candidate.slug))
            for e in enemy:
                if candidate.slug in enemy_synergy[e] or e in own_synergies:
                    score += ENEMY_SYNERGY
                    reasons.append(f"works with {self._name(e)}")
            own_victims = set(self.knowledge.countered_by(candidate.slug))
            for a in ally:
                if a in own_victims:
                    score += ALLY_COUNTERED
                    reasons.append(f"counters your {self._name(a)}")
            extra, why = self._setting(candidate.slug, map_slug)
            score += extra
            flagged = self._ban_flagged(candidate.slug, map_slug)
            if flagged:
                score += BAN_FLAG
                why = [*why, "recommended ban"]
            ranked.append(Suggestion(candidate, score, tuple(reasons + why)))
        return self._top(ranked, limit)

    # ── factors ──────────────────────────────────────────────────────────────

    def _setting(self, hero: str, map_slug: str | None) -> tuple[float, list[str]]:
        """Map rating and tier factors for ``hero`` (nothing when the data is missing)."""
        score, reasons = 0.0, []
        where = self.maps.get(map_slug or "", "")
        if map_slug:
            rating = self.knowledge.map_rating(hero, map_slug)
            if rating and MAP_RATING.get(rating):
                score += MAP_RATING[rating]
                reasons.append(f"{rating} on {where or map_slug}")
        rank = self.knowledge.tier(hero, map_slug) if map_slug else None
        label = where or "overall"
        if rank is None:
            rank = self.knowledge.tier(hero, "general")
            label = "overall"
        if rank and TIER_BONUS.get(rank.tier):
            score += TIER_BONUS[rank.tier]
            reasons.append(f"{rank.tier} tier {label}")
        return score, reasons

    def _ban_flagged(self, hero: str, map_slug: str | None) -> bool:
        for key in (map_slug, "general"):
            rank = self.knowledge.tier(hero, key) if key else None
            if rank is not None:
                return rank.ban
        return False

    def _name(self, slug: str) -> str:
        hero = self.roster.get(slug)
        return hero.name if hero else slug

    @staticmethod
    def _top(ranked: Iterable[Suggestion], limit: int) -> list[Suggestion]:
        """The best ``limit`` suggestions with a positive score (ties by name)."""
        useful = [s for s in ranked if s.score > 0]
        return sorted(useful, key=lambda s: (-s.score, s.hero.name))[:limit]
