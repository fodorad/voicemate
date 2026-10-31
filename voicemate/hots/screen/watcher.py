"""Draft mode: watches the screen and keeps the draft up to date.

The watcher looks at the screen only while Adam has switched game mode on (:meth:`start`); a
command that asks for a read is the only other time the screen is captured. While no draft is
on screen it takes a quick, rough look every few seconds. When a ban or pick screen appears it
starts a draft: the draft state is reset, a screen log folder is opened and from then on every
look is a full read that adds what the screen shows (locked heroes, bans, the map) to the
:class:`~voicemate.hots.draft.Draft`. "MATCH STARTING" ends the draft; the watcher then stays
quiet until the next draft screen shows up.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from voicemate.hots.draft import Draft
from voicemate.hots.roster import Roster
from voicemate.hots.screen.interpreter import BanReader, DraftScreen, FaceReader, interpret
from voicemate.hots.screen.layout import Calibration
from voicemate.hots.screen.log import ScreenLog
from voicemate.hots.screen.reader import FULL, Frame, ScreenError, ScreenReader
from voicemate.pipeline.skills import SkillAnswer

logger = logging.getLogger(__name__)

#: No draft seen yet (or it was abandoned): the watcher takes quick looks.
IDLE: str = "idle"
#: A draft is on screen: the watcher follows it.
DRAFT: str = "draft"
#: The loading screen shows both teams: the watcher noted them (and briefed Adam) and waits.
LOADING: str = "loading"
#: The match started: the watcher is quiet until the next draft screen.
OVER: str = "over"
#: A draft screen missing for this many looks in a row means the draft was abandoned.
LOST_AFTER: int = 5
#: After this many idle looks the watcher slows down (the game is probably not even open).
SLOW_AFTER: int = 30
SLOW_FACTOR: int = 5
#: Seconds to wait after the screen could not be read, before trying again.
PROBLEM_RETRY_S: float = 15.0
#: A full read this recent is reused by a command instead of reading again.
FRESH_S: float = 2.0
#: Without the watcher a draft read this long ago no longer counts as the draft in progress.
STALE_S: float = 600.0


class DraftWatcher:
    """Follows the draft on screen (a :class:`~voicemate.pipeline.skills.Background` worker).

    Args:
        reader: Looks at the screen.
        roster: Resolves hero names.
        maps: ``slug → name`` of the maps.
        draft: The session's draft, kept current while a draft is on screen.
        portraits: Reads the ban slots.
        faces: Cross-checks hero labels against the heroes' portraits.
        briefing: Says what the two teams mean for the match; called once when the loading
            screen shows them, and what it returns is said (nothing is said when it returns
            ``None``).
        calibration: Maps each capture onto the reference draft screen, whatever the machine's
            screen size or window position (the capture is read as it is when omitted).
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
        faces: FaceReader | None = None,
        briefing: Callable[[], str | None] | None = None,
        calibration: Calibration | None = None,
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
        self.faces = faces
        self.briefing = briefing
        self.calibration = calibration
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
        #: Whether game mode is on (the screen is watched; otherwise it is read on request only).
        self.watching = False
        self._read_at = float("-inf")
        self._lock = asyncio.Lock()
        self._woken = asyncio.Event()
        self._missed = 0
        self._idle_looks = 0
        self._logged: dict[str, Any] | None = None
        self._was_mine = False
        self._say: Callable[[SkillAnswer], Awaitable[None]] | None = None

    async def run(self, say: Callable[[SkillAnswer], Awaitable[None]]) -> None:
        """Watch until cancelled; ``say`` is used for the optional proactive cues."""
        self._say = say
        while True:
            await self._woken.wait()
            try:
                await self.poll()
            except Exception as exc:  # a bug or odd screen must not end the watching
                logger.exception("Draft look failed")
                self._fail(f"{type(exc).__name__}: {exc}")
            await self._sleep(self._interval())

    def start(self) -> None:
        """Switch game mode on: look at the screen every few seconds."""
        self.watching = True
        self._woken.set()

    def stop(self) -> None:
        """Switch game mode off: the screen is read only when asked."""
        self.watching = False
        self._woken.clear()
        self.end()

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
            screen = self._see(frame)
            if screen.is_loading:
                await self._loading(frame, screen)
                return
            if self.mode == LOADING and not screen.is_draft:
                self._missed += 1
                if self._missed >= LOST_AFTER:
                    logger.info("Loading screen gone: the match is on")
                    self.end()
                return
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
                screen = self._see(frame)
            await self._update(frame, screen)

    def end(self) -> None:
        """Stop following the draft (it is over); a later draft screen starts a new one."""
        self.mode = OVER
        self._idle_looks = 0

    async def look(self, reader: ScreenReader | None = None) -> DraftScreen:
        """A full read of the screen now (a recent one is reused); a draft screen joins the draft.

        Args:
            reader: Read this instead of the live screen (e.g. a saved screenshot); such a read
                is never reused.

        Raises:
            ScreenError: If the screen cannot be read.
        """
        async with self._lock:
            age = self._monotonic() - self._read_at
            if reader is None and self.screen is not None and self.mode == DRAFT:
                if age < FRESH_S:
                    return self.screen
                if not self.watching and age > STALE_S:
                    self.mode = IDLE
            frame = await self._capture(fast=False, reader=reader)
            screen = self._see(frame)
            if screen.is_loading:
                self._enter_loading(frame, screen)
            if self.mode != DRAFT and screen.is_draft:
                self._begin()
            if self.mode == DRAFT:
                await self._update(frame, screen)
            return screen

    # ── internals ────────────────────────────────────────────────────────────

    async def _capture(self, fast: bool, reader: ScreenReader | None = None) -> Frame:
        return await asyncio.to_thread((reader or self.reader).read, FULL, fast)

    def _see(self, frame: Frame) -> DraftScreen:
        """What the draft screen in ``frame`` shows."""
        if self.calibration is not None:
            frame = self.calibration.canonical(frame, self.maps)
        return interpret(frame, self.roster, self.maps, self.portraits, self.faces)

    def _fail(self, message: str) -> None:
        if message != self.problem:
            logger.warning("Cannot read the screen: %s", message)
        self.problem = message

    async def _loading(self, frame: Frame, screen: DraftScreen) -> None:
        """The loading screen: note the teams and brief Adam once (it is read in full first)."""
        if self.mode != LOADING:
            try:
                frame = await self._capture(fast=False)
            except ScreenError as exc:
                self._fail(str(exc))
                return
            screen = self._see(frame)
            if not screen.is_loading:
                return
        if self._enter_loading(frame, screen):
            await self._brief()

    def _enter_loading(self, frame: Frame, screen: DraftScreen) -> bool:
        """Take the teams of a loading screen into the draft; ``True`` when it just appeared."""
        fresh = self.mode != LOADING
        if fresh:
            logger.info("Loading screen: the match is about to start")
            if self.mode == IDLE:
                self.draft.reset()  # no draft was seen: these teams are all there is
            self.mode = LOADING
        self._missed = 0
        self._apply(screen)
        self.screen = screen
        self._read_at = self._monotonic()
        if fresh and self.log is not None:
            self.log.save(frame.image, self._state(screen))
        return fresh

    async def _brief(self) -> None:
        """Say what the teams mean for the match (when there is a briefing to give)."""
        text = self.briefing() if self.briefing is not None else None
        if text and self._say is not None:
            await self._say(SkillAnswer(text, "en"))

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

    @staticmethod
    def _state(screen: DraftScreen) -> dict[str, Any]:
        """What was read from a screen, as stored in the screen log."""
        return {
            "phase": screen.phase.kind,
            "mine": screen.phase.mine,
            "map": screen.map_slug,
            "ally": list(screen.ally),
            "pending": list(screen.pending),
            "enemy": list(screen.enemy),
            "banned": list(screen.banned),
        }

    def _record(self, frame: Frame, screen: DraftScreen) -> None:
        """Save the screen to the log when what it shows changed."""
        state = self._state(screen)
        if self.log is not None and state != self._logged:
            self.log.save(frame.image, {**state, "seconds": screen.seconds})
        self._logged = state

    async def _cue(self, screen: DraftScreen) -> None:
        """Tell Adam when it becomes his turn (only when ``proactive``)."""
        mine = screen.phase.mine
        if self.proactive and mine and not self._was_mine and self._say is not None:
            await self._say(SkillAnswer(f"Your {screen.phase.kind}.", "en"))
        self._was_mine = mine
