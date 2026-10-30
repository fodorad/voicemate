"""Spoken draft commands: rules that turn an utterance into advice, without the LLM.

:class:`Intents` recognizes a small English command language ("what's good against Zul'jin",
"enemy has Tychus and Muradin", "we're on Cursed Hollow", "what should I ban"), keeps the
draft in a :class:`~voicemate.hots.draft.Draft` and answers with one short templated sentence
per command. An utterance it does not recognize, or one that names nothing it knows, returns
``None`` so the chat model handles it as usual.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from voicemate.hots.advisor import Advisor, Suggestion
from voicemate.hots.aliases import AliasStore
from voicemate.hots.draft import Draft
from voicemate.hots.roster import Hero, normalize

_ENEMY = r"(?:the )?(?:enemy(?: team)?|enemies|they|opponents?|their team)"
_ALLY = r"(?:my team|our team|the team|allies|ally|we|i)"
_VERB = (
    r"(?:also |now )?(?:has|have|had|picked|took|got|are playing|is playing|plays?|"
    r"add|added|adding|locked in|lock in)"
)
_SIDES = r"enemy|enemies|opponents?|they|their team|ally|allies|my team|our team|team|we|i"

_RESET = re.compile(r"\b(?:reset|clear|restart)\b.*\bdraft\b|\bnew draft\b|\bstart over\b")
_MAP = re.compile(
    r"^(?:the )?map(?: is)? (?P<m>.+)$|^(?:we(?:'re| are)|i(?:'m| am)|playing|going|drafting) "
    r"(?:on|in) (?P<m2>.+)$"
)
_BANNED = re.compile(
    r"^(?:(?:they|we|enemy|the enemy|someone) )?banned (?P<h>.+)$|"
    r"^(?P<h2>.+?) (?:is|was|got) banned$"
)
_REMOVE = re.compile(
    rf"^(?:remove|delete|drop|take out) (?P<h>.+?) from (?:the )?(?P<side>{_SIDES})$|"
    rf"^(?P<side2>{_SIDES}) (?:dropped|removed) (?P<h2>.+)$"
)
_ADD = re.compile(
    rf"^(?:add|adding) (?P<h>.+?) to (?:the )?(?P<side>{_SIDES})$|"
    rf"^(?:{_ENEMY}|(?P<ally>{_ALLY})) {_VERB} (?P<h2>.+)$"
)
_BAN_ASK = re.compile(r"\bban(?:s|ning)?\b")
_SUGGEST = re.compile(r"^(?:suggest|suggestions?|recommend|recommendations?|advice)$")
_PICK_ASK = re.compile(r"\b(?:pick|picks|choose)\b")
_COUNTER_WORDS = re.compile(r"\b(?:against|versus|vs|counters?|counterpicks?|beats?)\b")
_FILLER = re.compile(r"^(?:(?:hey|ok|okay) )?(?:ava )?(?:please )?")
_NOISE = re.compile(r"\b(?:a|an|the|also|then|and then|please)\b")
_LIST_SPLIT = re.compile(r",| and | & ")

_ORDINALS = {
    "first": 0, "1st": 0, "one": 0, "yes": 0, "yeah": 0, "yep": 0,
    "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2,
}  # fmt: skip

#: How many heroes an answer names.
ANSWER_HEROES: int = 3


def _clean(text: str) -> str:
    lowered = text.lower().replace("’", "'")
    lowered = re.sub(r"[^a-z0-9', &-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,")
    return _FILLER.sub("", lowered).strip(" ,")


def _side(word: str) -> str:
    return (
        "enemy"
        if word in {"enemy", "enemies", "opponent", "opponents", "they", "their team"}
        else "ally"
    )


@dataclass(frozen=True)
class _Question:
    """A name that could be one of several heroes, waiting for Adam's pick."""

    said: str
    heard: str
    candidates: tuple[Hero, ...]


class Intents:
    """Recognizes draft commands and answers them.

    Args:
        advisor: Ranks counters, picks and bans.
        draft: The session's draft (a new one when omitted).
        aliases: Where spellings settled by a question are remembered.
    """

    def __init__(
        self, advisor: Advisor, draft: Draft | None = None, aliases: AliasStore | None = None
    ) -> None:
        self.advisor = advisor
        self.draft = draft if draft is not None else Draft()
        self.aliases = aliases
        self._said = ""
        self._question: _Question | None = None

    def handle(self, text: str) -> str | None:
        """Answer ``text`` when it is a draft command, otherwise return ``None``."""
        said = _clean(text)
        if not said:
            return None
        answered = self._answer_question(said)
        if answered is not None:
            return answered
        self._said = said
        for rule in (
            self._reset, self._map, self._banned, self._remove, self._add,
            self._ban_advice, self._suggest, self._pick_advice, self._counter_advice,
        ):  # fmt: skip
            answer = rule(said)
            if answer is not None:
                return answer
        return None

    def _answer_question(self, said: str) -> str | None:
        """Settle the pending "A or B?" with ``said``; any other speech drops the question."""
        question, self._question = self._question, None
        if question is None or question.heard not in question.said:
            return None
        if said in _ORDINALS and _ORDINALS[said] < len(question.candidates):
            hero = question.candidates[_ORDINALS[said]]
        else:
            hero = self.advisor.roster.resolve(said, {c.slug for c in question.candidates}).hero
        if hero is None:
            return None
        if self.advisor.roster.learn(question.heard, hero.slug) and self.aliases:
            self.aliases.add(question.heard, hero.slug)
        return self.handle(question.said.replace(question.heard, hero.name.lower(), 1))

    # ── commands that change the draft ───────────────────────────────────────

    def _reset(self, said: str) -> str | None:
        if not _RESET.search(said):
            return None
        self.draft.reset()
        return "Draft cleared."

    def _map(self, said: str) -> str | None:
        match = _MAP.match(said)
        if not match:
            return None
        slug = self._resolve_map(match.group("m") or match.group("m2"))
        if slug is None:
            return None
        self.draft.map_slug = slug
        return f"Map: {self.advisor.maps[slug]}."

    def _banned(self, said: str) -> str | None:
        match = _BANNED.match(said)
        if not match:
            return None
        return self._change("banned", match.group("h") or match.group("h2"), add=True)

    def _remove(self, said: str) -> str | None:
        match = _REMOVE.match(said)
        if not match:
            return None
        return self._change(
            _side(match.group("side")), match.group("h") or match.group("h2"), add=False
        )

    def _add(self, said: str) -> str | None:
        match = _ADD.match(said)
        if not match:
            return None
        if match.group("side"):
            side = _side(match.group("side"))
        else:
            side = "ally" if match.group("ally") else "enemy"
        return self._change(side, match.group("h") or match.group("h2"), add=True)

    def _change(self, side: str, fragment: str, add: bool) -> str | None:
        heroes, problems = self._heroes(fragment, self._available() if add else None)
        if not heroes and not problems:
            return None
        if heroes:
            slugs = [h.slug for h in heroes]
            if add:
                self.draft.add(side, slugs)
            else:
                self.draft.remove(side, slugs)
        if problems and not heroes:
            return problems[0]
        label = {"enemy": "Enemy team", "ally": "Your team", "banned": "Banned"}[side]
        names = ", ".join(self._name(s) for s in self.draft.side(side)) or "nobody"
        return f"{label}: {names}." + (f" {problems[0]}" if problems else "")

    # ── advice ───────────────────────────────────────────────────────────────

    def _ban_advice(self, said: str) -> str | None:
        if not _BAN_ASK.search(said):
            return None
        suggestions = self.advisor.bans(
            self.draft.enemy, self.draft.ally, self.draft.map_slug, self.draft.banned, ANSWER_HEROES
        )
        return self._sentence("Ban", suggestions, "I see no clear ban yet.")

    def _suggest(self, said: str) -> str | None:
        return self._picks(self.draft.enemy) if _SUGGEST.match(said) else None

    def _pick_advice(self, said: str) -> str | None:
        if not _PICK_ASK.search(said):
            return None
        against = _COUNTER_WORDS.search(said)
        enemy = self.draft.enemy
        if against:
            heroes, problems = self._heroes(said[against.end() :])
            if heroes:
                enemy = [h.slug for h in heroes]
            elif problems:
                return problems[0]
        return self._picks(enemy)

    def _counter_advice(self, said: str) -> str | None:
        words = list(_COUNTER_WORDS.finditer(said))
        if not words:
            return None
        heroes, problems = self._heroes(said[words[-1].end() :])
        if not heroes:
            if problems:
                return problems[0]
            return (
                self._picks(self.draft.enemy)
                if said.endswith(("counterpick", "counterpicks"))
                else None
            )
        if len(heroes) == 1:
            hero = heroes[0]
            suggestions = self.advisor.counters(hero.slug, self.draft.map_slug, ANSWER_HEROES)
            return self._sentence(
                f"To counter {hero.name}",
                suggestions,
                f"I have no counters listed for {hero.name}.",
            )
        return self._picks([h.slug for h in heroes])

    def _picks(self, enemy: Sequence[str]) -> str:
        suggestions = self.advisor.picks(
            enemy, self.draft.ally, self.draft.map_slug, self.draft.banned, ANSWER_HEROES
        )
        return self._sentence("Pick", suggestions, "I see no clear pick yet.")

    # ── names ────────────────────────────────────────────────────────────────

    def _available(self) -> set[str]:
        """Heroes that can still be picked or banned."""
        taken = {*self.draft.enemy, *self.draft.ally, *self.draft.banned}
        return {hero.slug for hero in self.advisor.roster} - taken

    def _heroes(
        self, fragment: str, available: set[str] | None = None
    ) -> tuple[list[Hero], list[str]]:
        """Heroes named in ``fragment`` and messages about names that were not clear.

        ``available`` limits sound-alike guesses to heroes not yet in the draft.
        """
        heroes: list[Hero] = []
        problems: list[str] = []
        for part in _LIST_SPLIT.split(_NOISE.sub(" ", fragment)):
            name = part.strip(" ,")
            if not name:
                continue
            resolved = self.advisor.roster.resolve(name, available)
            if resolved.hero is not None:
                if resolved.hero not in heroes:
                    heroes.append(resolved.hero)
            elif resolved.candidates:
                options = " or ".join(h.name for h in resolved.candidates)
                problems.append(f"{options}?")
                if not problems[1:] and len(resolved.candidates) > 1:
                    self._question = _Question(self._said, name, resolved.candidates)
        return heroes, problems

    def _resolve_map(self, spoken: str) -> str | None:
        key = normalize(spoken)
        if len(key) < 3:
            return None
        names = {slug: normalize(name) for slug, name in self.advisor.maps.items()}
        found = [slug for slug, name in names.items() if key in name]
        if not found:
            ratios = {
                slug: difflib.SequenceMatcher(None, key, name).ratio()
                for slug, name in names.items()
            }
            found = [
                slug
                for slug, ratio in ratios.items()
                if ratio >= 0.75 and ratio == max(ratios.values())
            ]
        return found[0] if len(found) == 1 else None

    def _name(self, slug: str) -> str:
        hero = self.advisor.roster.get(slug)
        return hero.name if hero else slug

    @staticmethod
    def _sentence(lead: str, suggestions: list[Suggestion], empty: str) -> str:
        if not suggestions:
            return empty
        names = ", ".join(s.hero.name for s in suggestions)
        first = suggestions[0]
        why = " and ".join(first.reasons[:2])
        return f"{lead}: {names}. {first.hero.name}: {why}." if why else f"{lead}: {names}."
