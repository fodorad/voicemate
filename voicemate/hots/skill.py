"""The HotS skill: opens the knowledge base once and gives each session its own draft."""

from __future__ import annotations

import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from voicemate.config import Config, ScreenConfig
from voicemate.hots.advisor import Advisor
from voicemate.hots.aliases import AliasStore
from voicemate.hots.briefing import Briefing
from voicemate.hots.commands import Command, parse
from voicemate.hots.draft import Draft
from voicemate.hots.intents import Intents, clean
from voicemate.hots.knowledge import Knowledge, RefreshReport
from voicemate.hots.screen.faces import Faces
from voicemate.hots.screen.interpreter import DraftScreen
from voicemate.hots.screen.layout import Calibration
from voicemate.hots.screen.log import ScreenLog
from voicemate.hots.screen.portraits import Portraits
from voicemate.hots.screen.reader import ScreenError, ScreenReader
from voicemate.hots.screen.watcher import OVER, DraftWatcher
from voicemate.pipeline.skills import Skill, SkillAnswer

logger = logging.getLogger(__name__)

#: Said when draft mode did not understand (short, so Adam can repeat at once).
SAY_AGAIN = "Say again."
#: Said when the screen cannot be captured.
BLIND = "I can't see the screen. Allow Screen Recording for this app."


@dataclass
class ScreenKit:
    """What reading the screen needs, shared by every session.

    Attributes:
        reader: Looks at the live screen.
        portraits: Reads the ban slots (``None`` when no portraits were saved).
        settings: The ``[hots.screen]`` configuration.
        saved: Reads the newest saved screenshot instead (``read the last screenshot``).
        calibration: Finds where the draft screen sits in the captured picture.
        faces: The learned pick portraits that cross-check the hero labels.
    """

    reader: ScreenReader
    portraits: Portraits | None
    settings: ScreenConfig
    saved: ScreenReader | None = None
    calibration: Calibration | None = None
    faces: Faces | None = None


class HotsSkill:
    """One session's draft commands (a :class:`~voicemate.pipeline.skills.Skill`).

    After "draft mode on" (or "game mode on") the skill is in *draft mode*: the screen is
    watched, answers are as short as possible and speech that is not a draft command gets
    "Say again." instead of going to the chat model. The match starting pauses the focus until
    the next draft or loading screen. Outside draft mode the screen is read only on request.

    Args:
        intents: The session's draft and command rules.
        watcher: Follows the draft on screen (``None`` when the screen cannot be read).
        saved: Reads the newest saved screenshot.
        on: Start in draft mode.
        briefing: Says what the two teams mean for the match (the loading screen, "brief").
    """

    def __init__(
        self,
        intents: Intents,
        watcher: DraftWatcher | None = None,
        saved: ScreenReader | None = None,
        on: bool = False,
        briefing: Briefing | None = None,
    ) -> None:
        self.intents = intents
        self.watcher = watcher
        self.saved = saved
        self.briefing = briefing
        self._on = False
        if on:
            self._switch(True)

    @property
    def draft_mode(self) -> bool:
        """Whether answers are terse and only draft speech is handled."""
        return self._on and (self.watcher is None or self.watcher.mode != OVER)

    def _switch(self, on: bool) -> None:
        self._on = on
        if self.watcher is not None:
            self.watcher.start() if on else self.watcher.stop()

    async def run(self, say: Callable[[SkillAnswer], Awaitable[None]]) -> None:
        """Watch the screen for drafts while the session is open (a ``Background`` skill)."""
        if self.watcher is not None:
            await self.watcher.run(say)

    async def answer(self, text: str) -> SkillAnswer | None:
        """Answer a draft command in English, or ``None`` for anything else."""
        said = clean(text)
        command = parse(said)
        if command is not None:
            return SkillAnswer(await self._command(command), "en")
        focused = self.draft_mode
        self.intents.terse = focused
        if focused:
            await self._refresh()
        answer = self.intents.handle(text)
        if answer is None and focused and said:
            answer = SAY_AGAIN
        return None if answer is None else SkillAnswer(answer, "en")

    # ── commands ─────────────────────────────────────────────────────────────

    async def _command(self, command: Command) -> str:
        if command.kind in ("on", "off"):
            self._switch(command.kind == "on")
            return f"Draft mode {command.kind}."
        if command.kind == "brief":
            await self._refresh()
            return self._brief()
        if self.watcher is None:
            return "I can't read the screen here."
        reader = self.saved if command.kind == "last" else None
        try:
            screen = await self.watcher.look(reader)
        except ScreenError as exc:
            logger.warning("Screen read failed: %s", exc)
            return str(exc) if command.kind == "last" else BLIND
        self.intents.terse = True
        return self._readback(screen, command.advice)

    def _brief(self) -> str:
        """What the teams in the draft mean for the match."""
        draft = self.intents.draft
        if self.briefing is None:
            return "I have no briefing here."
        return self.briefing.say(draft.ally, draft.enemy, draft.map_slug)

    def _readback(self, screen: DraftScreen, wanted: str) -> str:
        """What the screen shows, then the advice that fits the phase."""
        draft = self.intents.draft
        if not screen.is_draft and not draft.enemy and not draft.ally:
            return "I see no draft."
        names = self.intents.name
        parts = []
        if draft.map_slug:
            parts.append(f"Map {self.intents.advisor.maps[draft.map_slug]}.")
        for label, heroes in (("Team", draft.ally), ("Enemy", draft.enemy), ("Bans", draft.banned)):
            if heroes:
                parts.append(f"{label} {', '.join(names(h) for h in heroes)}.")
        if screen.is_loading and self.briefing is not None:
            parts.append(self._brief())
        kind = wanted or (screen.phase.kind if screen.phase.mine else "")
        if kind in ("ban", "pick"):
            advice = self.intents.handle(kind)
            if advice:
                parts.append(advice)
        return " ".join(parts) or "I see no draft."

    async def _refresh(self) -> None:
        """Bring the draft up to date with the screen (a read this recent is reused)."""
        if self.watcher is None:
            return
        try:
            await self.watcher.look()
        except ScreenError as exc:
            logger.warning("Screen read failed: %s", exc)


class Hots:
    """The opened HotS knowledge, shared by all sessions.

    Args:
        knowledge: The imported pages.
        report: What the startup refresh changed.
        aliases: Spellings learned from Adam.
        screen: Reading the screen (``None`` when it is off or not supported).
    """

    def __init__(
        self,
        knowledge: Knowledge,
        report: RefreshReport,
        aliases: AliasStore | None = None,
        screen: ScreenKit | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.report = report
        self.aliases = aliases
        self.screen = screen
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
        hots = cls(knowledge, report, AliasStore(config.hots.aliases_path), _screen_kit(config))
        if not len(hots.advisor.roster):
            logger.info("No HotS pages in %s; the draft skill is off", config.hots.pages_path)
            knowledge.close()
            return None
        logger.info("HotS skill ready: %d heroes", len(hots.advisor.roster))
        return hots

    def skill(self) -> HotsSkill:
        """A skill with a fresh, empty draft for one session."""
        draft = Draft()
        intents = Intents(self.advisor, draft, aliases=self.aliases)
        briefing = Briefing(self.advisor)
        if self.screen is None:
            return HotsSkill(intents, briefing=briefing)
        settings = self.screen.settings
        watcher = DraftWatcher(
            self.screen.reader,
            self.advisor.roster,
            self.advisor.maps,
            draft,
            portraits=self.screen.portraits,
            faces=self.screen.faces,
            briefing=(lambda: briefing.say(draft.ally, draft.enemy, draft.map_slug))
            if settings.loading_briefing
            else None,
            calibration=self.screen.calibration,
            log=ScreenLog(settings.log_path, settings.keep),
            poll_s=settings.poll_s,
            proactive=settings.proactive,
        )
        return HotsSkill(
            intents, watcher, self.screen.saved, on=settings.game_mode, briefing=briefing
        )

    def session_skills(self) -> list[Skill]:
        """The skills of one new session (what :attr:`Runtime.skills` calls)."""
        return [self.skill()]

    def close(self) -> None:
        """Close the knowledge base."""
        self.knowledge.close()


def _screen_kit(config: Config) -> ScreenKit | None:
    """The screen reading setup, or ``None`` when it is disabled or the platform cannot do it."""
    settings = config.hots.screen
    if not settings.enabled or sys.platform != "darwin":
        return None
    return _mac_kit(config)  # pragma: no cover - needs macOS


def _mac_kit(config: Config) -> ScreenKit:  # pragma: no cover - needs macOS and a screen
    from voicemate.hots.screen.macos import ImageFileReader, MacScreenReader

    settings = config.hots.screen
    images = config.hots.images_path
    portraits = Portraits.from_folder(images) if images.is_dir() else None
    if portraits is None or not portraits.slugs:
        logger.info("No hero portraits in %s: bans are read from the announcements only", images)
        portraits = None
    log = ScreenLog(settings.log_path, settings.keep)
    calibration = config.data_path / "calibration"

    def newest():
        found = log.newest_screenshot()
        if found is not None:
            return found
        shots = sorted(calibration.glob("*.png")) if calibration.is_dir() else []
        return shots[-1] if shots else None

    return ScreenKit(
        MacScreenReader(settings.window),
        portraits,
        settings,
        ImageFileReader(newest),
        Calibration(settings.calibration_path),
        Faces(settings.faces_path),
    )
