"""The HotS skill: opens the knowledge base once and gives each session its own draft."""

from __future__ import annotations

import logging

from voicemate.config import Config
from voicemate.hots.advisor import Advisor
from voicemate.hots.aliases import AliasStore
from voicemate.hots.intents import Intents
from voicemate.hots.knowledge import Knowledge, RefreshReport
from voicemate.pipeline.skills import Skill, SkillAnswer

logger = logging.getLogger(__name__)


class HotsSkill:
    """One session's draft commands (a :class:`~voicemate.pipeline.skills.Skill`).

    Args:
        intents: The session's draft and command rules.
    """

    def __init__(self, intents: Intents) -> None:
        self.intents = intents

    def answer(self, text: str) -> SkillAnswer | None:
        """Answer a draft command in English, or ``None`` for anything else."""
        answer = self.intents.handle(text)
        return None if answer is None else SkillAnswer(answer, "en")


class Hots:
    """The opened HotS knowledge, shared by all sessions.

    Args:
        knowledge: The imported pages.
        report: What the startup refresh changed.
    """

    def __init__(
        self, knowledge: Knowledge, report: RefreshReport, aliases: AliasStore | None = None
    ) -> None:
        self.knowledge = knowledge
        self.report = report
        self.aliases = aliases
        self.advisor = Advisor(knowledge)
        for spelling, slug in (aliases.load() if aliases else {}).items():
            self.advisor.roster.learn(spelling, slug)

    @classmethod
    def open(cls, config: Config) -> Hots | None:
        """Open the knowledge base and import new or changed pages (startup only).

        Returns:
            ``None`` when the skill is disabled or no hero pages have been saved yet.
        """
        if not config.hots.enabled:
            return None
        knowledge = Knowledge(config.hots_database_path, config.hots.pages_path)
        report = knowledge.refresh()
        hots = cls(knowledge, report, AliasStore(config.hots.aliases_path))
        if not len(hots.advisor.roster):
            logger.info("No HotS pages in %s; the draft skill is off", config.hots.pages_path)
            knowledge.close()
            return None
        logger.info("HotS skill ready: %d heroes", len(hots.advisor.roster))
        return hots

    def skill(self) -> HotsSkill:
        """A skill with a fresh, empty draft for one session."""
        return HotsSkill(Intents(self.advisor, aliases=self.aliases))

    def session_skills(self) -> list[Skill]:
        """The skills of one new session (what :attr:`Runtime.skills` calls)."""
        return [self.skill()]

    def close(self) -> None:
        """Close the knowledge base."""
        self.knowledge.close()
