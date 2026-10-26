"""The hero roster and name resolution (typed or misheard by speech recognition)."""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

#: Spoken spellings that do not normalize to the hero's name, as ``spelling → slug``.
SPOKEN_ALIASES: dict[str, str] = {
    "lee ming": "li-ming",
    "lily": "li-li",
    "lilly": "li-li",
    "etc": "e-t-c",
    "e t c": "e-t-c",
    "vikings": "the-lost-vikings",
    "lost vikings": "the-lost-vikings",
    "tlv": "the-lost-vikings",
    "diva": "dva",
    "d va": "dva",
    "sonia": "sonya",
    "cho gall": "cho",
    "chogall": "cho",
    "butcher": "the-butcher",
    "lieutenant morales": "lt-morales",
    "morales": "lt-morales",
    "sergeant hammer": "sgt-hammer",
    "hammer": "sgt-hammer",
    "kel thuzad": "kel-thuzad",
    "kelthuzad": "kel-thuzad",
    "gul dan": "guldan",
    "mal ganis": "malganis",
    "zul jin": "zuljin",
    "anub arak": "anubarak",
    "anubis": "anubarak",
    "tosh": "raynor",
}

#: Similarity (0-1) a spelling needs to count as a match, and the margin over the runner-up.
MATCH_THRESHOLD: float = 0.8
MATCH_MARGIN: float = 0.05
#: A shorter spelling is never matched as the start of a name.
MIN_PREFIX: int = 3
#: Weakest similarity still offered as a "did you mean" candidate.
SUGGEST_THRESHOLD: float = 0.6


def normalize(text: str) -> str:
    """Lower-case ``text`` without accents, punctuation or spaces (``Zul'jin`` → ``zuljin``)."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain.lower())


@dataclass(frozen=True)
class Hero:
    """One hero of the roster.

    Attributes:
        slug: Icy Veins slug (``li-ming``), the hero's identity everywhere.
        name: Display name (``Li-Ming``).
        role: Role such as ``Tank`` or ``Ranged Assassin``; empty when unknown.
    """

    slug: str
    name: str
    role: str = ""


@dataclass(frozen=True)
class Resolved:
    """The outcome of resolving a spoken or typed hero name.

    Attributes:
        hero: The hero when the name is unambiguous, otherwise ``None``.
        candidates: Closest heroes when the name is ambiguous or unknown (best first).
    """

    hero: Hero | None
    candidates: tuple[Hero, ...] = ()


class Roster:
    """All heroes, with name resolution.

    Args:
        heroes: The heroes.
        aliases: Extra spellings as ``spelling → slug`` (default: :data:`SPOKEN_ALIASES`).
    """

    def __init__(self, heroes: Iterable[Hero], aliases: Mapping[str, str] | None = None) -> None:
        self._by_slug = {hero.slug: hero for hero in heroes}
        self._keys: dict[str, Hero] = {}
        for hero in self._by_slug.values():
            self._keys[normalize(hero.name)] = hero
            self._keys[normalize(hero.slug)] = hero
        for spelling, slug in (SPOKEN_ALIASES if aliases is None else aliases).items():
            if slug in self._by_slug:
                self._keys[normalize(spelling)] = self._by_slug[slug]

    def __len__(self) -> int:
        """Number of heroes."""
        return len(self._by_slug)

    def __iter__(self) -> Iterator[Hero]:
        """Iterate over the heroes."""
        return iter(self._by_slug.values())

    def get(self, slug: str) -> Hero | None:
        """The hero with this slug, if any."""
        return self._by_slug.get(slug)

    def resolve(self, text: str) -> Resolved:
        """Find the hero ``text`` names.

        An exact (normalized) name, slug or alias wins; so does the start of exactly one name
        (``mura``); otherwise the best fuzzy match counts when it clearly beats every other hero.
        Anything else returns the closest candidates.
        """
        key = normalize(text)
        if not key:
            return Resolved(None)
        if key in self._keys:
            return Resolved(self._keys[key])
        if len(key) >= MIN_PREFIX:
            starting = {h.slug: h for k, h in self._keys.items() if k.startswith(key)}
            if len(starting) == 1:
                return Resolved(next(iter(starting.values())))
            if starting:
                return Resolved(None, tuple(starting.values())[:3])
        best: dict[str, tuple[float, Hero]] = {}
        for candidate, hero in self._keys.items():
            ratio = difflib.SequenceMatcher(None, key, candidate).ratio()
            if ratio > best.get(hero.slug, (0.0, hero))[0]:
                best[hero.slug] = (ratio, hero)
        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        close = tuple(hero for ratio, hero in ranked[:3] if ratio >= SUGGEST_THRESHOLD)
        if ranked and ranked[0][0] >= MATCH_THRESHOLD:
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if ranked[0][0] - runner_up >= MATCH_MARGIN:
                return Resolved(ranked[0][1])
        return Resolved(None, close)
