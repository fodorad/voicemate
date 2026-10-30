"""The hero roster and name resolution (typed or misheard by speech recognition)."""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Collection, Iterable, Iterator, Mapping
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


#: Letter groups that sound alike, replaced in this order before the vowels are dropped.
_SOUNDS: tuple[tuple[str, str], ...] = (
    ("ph", "f"), ("th", "t"), ("sh", "S"), ("ch", "S"), ("ck", "k"), ("x", "ks"), ("q", "k"),
    ("c", "k"), ("z", "s"), ("w", "v"), ("y", "i"),
)  # fmt: skip
#: Sound keys shorter than this say too little to match by.
MIN_SOUND_KEY: int = 3


def normalize(text: str) -> str:
    """Lower-case ``text`` without accents, punctuation or spaces (``Zul'jin`` → ``zuljin``)."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain.lower())


def sound_key(text: str) -> str:
    """A rough sound-alike key: the consonant skeleton of how ``text`` is pronounced.

    ``Kszalatat`` and ``Xal'atath`` both become ``kslt``, so a badly pronounced name still
    lands next to the right hero.
    """
    key = normalize(text)
    for letters, sound in _SOUNDS:
        key = key.replace(letters, sound)
    key = re.sub(r"[aeiouh0-9]", "", key)
    return re.sub(r"(.)\1+", r"\1", key)


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


def _similarity(spelled: str, candidate: str, heard: str) -> float:
    """How well a spelling and its sound key fit a candidate name (0-1)."""
    ratio = difflib.SequenceMatcher(None, spelled, candidate).ratio()
    sound = sound_key(candidate)
    if len(heard) < MIN_SOUND_KEY or len(sound) < MIN_SOUND_KEY:
        return ratio
    return (ratio + difflib.SequenceMatcher(None, heard, sound).ratio()) / 2


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

    def learn(self, spelling: str, slug: str) -> bool:
        """Make ``spelling`` mean the hero ``slug``; ``False`` when it already means something."""
        key = normalize(spelling)
        if slug not in self._by_slug or key in self._keys:
            return False
        self._keys[key] = self._by_slug[slug]
        return True

    def get(self, slug: str) -> Hero | None:
        """The hero with this slug, if any."""
        return self._by_slug.get(slug)

    def resolve(self, text: str, available: Collection[str] | None = None) -> Resolved:
        """Find the hero ``text`` names.

        An exact (normalized) name, slug or alias wins; so does the start of exactly one name
        (``mura``); otherwise the best match by spelling and sound counts when it clearly beats
        every other hero. Anything else returns the closest candidates.

        Args:
            text: The typed or recognized name.
            available: Slugs of the heroes that can still be meant; the start-of-name and
                sound-alike matches look only at these (an exact name always wins).
        """
        key = normalize(text)
        if not key:
            return Resolved(None)
        if key in self._keys:
            return Resolved(self._keys[key])
        keys = {k: h for k, h in self._keys.items() if available is None or h.slug in available}
        if len(key) >= MIN_PREFIX:
            starting = {h.slug: h for k, h in keys.items() if k.startswith(key)}
            if len(starting) == 1:
                return Resolved(next(iter(starting.values())))
            if starting:
                return Resolved(None, tuple(starting.values())[:3])
        best: dict[str, tuple[float, Hero]] = {}
        heard = sound_key(text)
        for candidate, hero in keys.items():
            ratio = _similarity(key, candidate, heard)
            if ratio > best.get(hero.slug, (0.0, hero))[0]:
                best[hero.slug] = (ratio, hero)
        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        close = tuple(hero for ratio, hero in ranked[:3] if ratio >= SUGGEST_THRESHOLD)
        if ranked and ranked[0][0] >= MATCH_THRESHOLD:
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if ranked[0][0] - runner_up >= MATCH_MARGIN:
                return Resolved(ranked[0][1])
        return Resolved(None, close)


def resolve_map(spoken: str, maps: Mapping[str, str]) -> str | None:
    """The slug of the map ``spoken`` names (``maps`` is ``slug → name``), if exactly one fits.

    A part of a name counts (``temple`` alone fits several maps and so names none); otherwise
    a close spelling counts when it is clearly the best.
    """
    key = normalize(spoken)
    if len(key) < MIN_PREFIX:
        return None
    names = {slug: normalize(name) for slug, name in maps.items()}
    found = [slug for slug, name in names.items() if key in name]
    if not found:
        ratios = {
            slug: difflib.SequenceMatcher(None, key, name).ratio() for slug, name in names.items()
        }
        best = max(ratios.values(), default=0.0)
        found = [slug for slug, ratio in ratios.items() if ratio >= 0.75 and ratio == best]
    return found[0] if len(found) == 1 else None
