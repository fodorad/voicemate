"""Skills: specialist answers that bypass the chat model.

A skill recognizes the questions it can answer exactly (from local data, in milliseconds) and
answers them in one sentence; everything else returns ``None`` and goes to the agent. The
session asks its skills before every ordinary turn and speaks the answer through the usual
pipeline (text-to-speech, UI, metrics), so a skill only has to produce text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SkillAnswer:
    """A skill's answer.

    Attributes:
        text: The sentence to show and speak.
        lang: Language of the sentence (picks the voice).
    """

    text: str
    lang: str


class Skill(Protocol):
    """Answers the questions it knows without the chat model."""

    def answer(self, text: str) -> SkillAnswer | None:
        """The answer to ``text``, or ``None`` when this skill does not handle it."""
        ...


def first_answer(skills: Iterable[Skill], text: str) -> SkillAnswer | None:
    """The first skill answer to ``text`` (skills are tried in order), or ``None``."""
    for skill in skills:
        answer = skill.answer(text)
        if answer is not None:
            return answer
    return None
