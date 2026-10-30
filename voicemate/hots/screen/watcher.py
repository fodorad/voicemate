"""Draft mode: watches the screen and keeps the draft up to date.

The watcher is idle while no draft is on screen (a quick, rough look every few seconds). When a
ban or pick screen appears it starts a draft: the draft state is reset, a screen log folder is
opened and from then on every look is a full read that adds what the screen shows (locked
heroes, bans, the map) to the :class:`~voicemate.hots.draft.Draft`. "MATCH STARTING" ends the
draft; the watcher then stays quiet until the next draft screen shows up.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from voicemate.hots.draft import Draft
from voicemate.hots.roster import Roster
from voicemate.hots.screen.interpreter import BanReader, DraftScreen, interpret
from voicemate.hots.screen.log import ScreenLog
from voicemate.hots.screen.reader import FULL, Frame, ScreenError, ScreenReader
from voicemate.pipeline.skills import SkillAnswer

logger = logging.getLogger(__name__)

#: Watcher modes.
IDLE, DRAFT, OVER = "idle", "draft", "over"
#: A draft screen missing for this many looks in a row means the draft was abandoned.
LOST_AFTER: int = 5
#: After this many idle looks the watcher slows down (the game is probably not even open).
SLOW_AFTER: int = 30
SLOW_FACTOR: int = 5
#: Seconds to wait after the screen could not be read, before trying again.
PROBLEM_RETRY_S: float = 15.0
#: A full read this recent is reused by a command instead of reading again.
FRESH_S: float = 2.0


class DraftWatcher:
    """Follows the draft on screen (a :class:`~voicemate.pipeline.skills.Background` worker).

    Args:
        reader: Looks at the screen.
        roster: Resolves hero names.
        maps: ``slug → name`` of the maps.
        draft: The session's draft, kept current while a draft is on screen.
        portraits: Reads the ban slots.
        log: Where captured screens are saved (not saved when omitted).
        poll_s: Seconds between looks.
        proactive: Say "Your ban." / "Your pick." when it becomes Adam's turn.
        sleep: Waits between looks (replaceable for tests).
        monotonic: Clock for the age of a read (replaceable for tests).
    """

    def __init__(
        self,
        reader: ScreenReader,
        roster: Roster,
        maps: Mapping[str, str],
        draft: Draft,
        portraits: BanReader | None = None,
        log: ScreenLog | None = None,
        poll_s: float = 2.0,
        proactive: bool = False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.reader = reader
        self.roster = roster
        self.maps = maps
        self.draft = draft
        self.portraits = portraits
        self.log = log
        self.poll_s = poll_s
        self.proactive = proactive
        self._sleep = sleep
        self._monotonic = monotonic
        #: ``"idle"`` (no draft seen), ``"draft"`` (one is on screen) or ``"over"`` (it ended).
        self.mode = IDLE
        #: The latest full read of the draft screen.
        self.screen: DraftScreen | None = None
        #: What went wrong with the last look (``None`` when it worked).
        self.problem: str | None = None
        self._read_at = float("-inf")
        self._lock = asyncio.Lock()
        self._missed = 0
        self._idle_looks = 0
        self._logged: dict[str, Any] | None = None
        self._was_mine = False
        self._say: Callable[[SkillAnswer], Awaitable[None]] | None = None

    async def run(self, say: Callable[[SkillAnswer], Awaitable[None]]) -> None:
        """Watch until cancelled; ``say`` is used for the optional proactive cues."""
        self._say = say
        while True:
            await self.poll()
            await self._sleep(self._interval())

    def _interval(self) -> float:
        if self.problem:
            return PROBLEM_RETRY_S
        if self.mode != DRAFT and self._idle_looks >= SLOW_AFTER:
            return self.poll_s * SLOW_FACTOR
        return self.poll_s

    async def poll(self) -> None:
        """One look: rough while waiting for a draft, full during one."""
        async with self._lock:
            try:
                frame = await self._capture(fast=self.mode != DRAFT)
            except ScreenError as exc:
                self._fail(str(exc))
                return
            self.problem = None
            screen = interpret(frame, self.roster, self.maps, self.portraits)
            if self.mode != DRAFT:
                if not screen.is_draft:
                    self._idle_looks += 1
                    return
                self._begin()
                try:
                    frame = await self._capture(fast=False)
                except ScreenError as exc:
                    self._fail(str(exc))
                    return
                screen = interpret(frame, self.roster, self.maps, self.portraits)
            await self._update(frame, screen)

    async def look(self) -> DraftScreen:
        """A full read of the screen now (a recent one is reused); a draft screen joins the draft.

        Raises:
            ScreenError: If the screen cannot be read.
        """
        async with self._lock:
            if (
                self.screen is not None
                and self.mode == DRAFT
                and self._monotonic() - self._read_at < FRESH_S
            ):
                return self.screen
            frame = await self._capture(fast=False)
            screen = interpret(frame, self.roster, self.maps, self.portraits)
            if self.mode != DRAFT and screen.is_draft:
                self._begin()
            if self.mode == DRAFT:
                await self._update(frame, screen)
            return screen

    # ── internals ────────────────────────────────────────────────────────────

    async def _capture(self, fast: bool) -> Frame:
        return await asyncio.to_thread(self.reader.read, FULL, fast)

    def _fail(self, message: str) -> None:
        if message != self.problem:
            logger.warning("Cannot read the screen: %s", message)
        self.problem = message

    def _begin(self) -> None:
        """A draft screen appeared: forget the old draft and open a new log folder."""
        logger.info("Draft started")
        self.mode = DRAFT
        self.draft.reset()
        self.screen = None
        self._missed = 0
        self._logged = None
        self._was_mine = False
        if self.log is not None:
            self.log.start_draft()

    async def _update(self, frame: Frame, screen: DraftScreen) -> None:
        if screen.phase.kind == "starting":
            logger.info("Match starting: the draft is over")
            self._record(frame, screen)
            self.mode = OVER
            self._idle_looks = 0
            return
        if not screen.is_draft:
            self._missed += 1
            if self._missed >= LOST_AFTER:
                logger.info("Draft screen gone: back to waiting")
                self.mode = IDLE
                self._idle_looks = 0
            return
        self._missed = 0
        self._apply(screen)
        self.screen = screen
        self._read_at = self._monotonic()
        self._record(frame, screen)
        await self._cue(screen)

    def _apply(self, screen: DraftScreen) -> None:
        """Add what the screen shows to the draft (locked heroes never unlock)."""
        draft = self.draft
        draft.map_slug = screen.map_slug or draft.map_slug
        draft.add("enemy", list(screen.enemy))
        draft.add("banned", list(screen.banned))
        draft.add("ally", [h for h in screen.ally if h not in draft.enemy + draft.banned])
        draft.remove("ally", [*draft.enemy, *draft.banned])

    def _record(self, frame: Frame, screen: DraftScreen) -> None:
        """Save the screen to the log when what it shows changed."""
        state = {
            "phase": screen.phase.kind,
            "mine": screen.phase.mine,
            "map": screen.map_slug,
            "ally": list(screen.ally),
            "pending": list(screen.pending),
            "enemy": list(screen.enemy),
            "banned": list(screen.banned),
        }
        if self.log is not None and state != self._logged:
            self.log.save(frame.image, {**state, "seconds": screen.seconds})
        self._logged = state

    async def _cue(self, screen: DraftScreen) -> None:
        """Tell Adam when it becomes his turn (only when ``proactive``)."""
        mine = screen.phase.mine
        if self.proactive and mine and not self._was_mine and self._say is not None:
            await self._say(SkillAnswer(f"Your {screen.phase.kind}.", "en"))
        self._was_mine = mine
