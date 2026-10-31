import asyncio
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tests.helpers import ScriptedScreen
from tests.hots.screen.screens import boxes
from tests.hots.screen.test_interpreter import HEROES, MAPS, screen_of
from tests.hots.screen.test_layout import place
from voicemate.hots.draft import Draft
from voicemate.hots.screen.layout import Calibration, Layout
from voicemate.hots.screen.log import ScreenLog
from voicemate.hots.screen.reader import FULL, ScreenError, TextBox
from voicemate.hots.screen.watcher import (
    DRAFT,
    FRESH_S,
    IDLE,
    LOADING,
    LOST_AFTER,
    OVER,
    PROBLEM_RETRY_S,
    SLOW_AFTER,
    SLOW_FACTOR,
    STALE_S,
    DraftWatcher,
)

ENEMY_ARTANIS = TextBox("ARTANIS", 1.0, 0.95, 0.77, 0.045, 0.06)


def shown(name: str, *extra: TextBox):
    return screen_of(*boxes(name), *extra)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class BrokenOnce:
    """A screen that fails with an unexpected error on the first read, then works."""

    def __init__(self, screen) -> None:
        self.screen = screen
        self.failed = False

    def read(self, region=FULL, fast=False):
        if not self.failed:
            self.failed = True
            raise RuntimeError("odd screen")
        return self.screen.read(region, fast)


class WatcherCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name) / "game_screens"
        self.clock = Clock()
        self.draft = Draft()
        self.said: list = []

    def watch(self, *frames, proactive=False, log=True) -> DraftWatcher:
        self.screen = ScriptedScreen(*frames)
        self.log = ScreenLog(self.folder, clock=lambda: datetime(2026, 10, 30, 20, 12, 7))
        watcher = DraftWatcher(
            self.screen,
            HEROES,
            MAPS,
            self.draft,
            log=self.log if log else None,
            proactive=proactive,
            monotonic=self.clock,
        )

        async def say(answer):
            self.said.append(answer.text)

        watcher._say = say
        return watcher


class TestLoading(WatcherCase):
    def briefed(self, *frames, text: str | None = "Plan."):
        watcher = self.watch(*frames)
        watcher.briefing = None if text is None else lambda: text
        return watcher

    async def test_the_loading_screen_is_read_in_full_and_briefed_once(self):
        watcher = self.briefed(shown("match_loading"))
        await watcher.poll()
        self.assertEqual(watcher.mode, LOADING)
        self.assertEqual(self.screen.reads, [(FULL, True), (FULL, False)])
        self.assertEqual(self.draft.ally, ["thrall", "auriel", "tychus", "jaina", "johanna"])
        self.assertEqual(self.draft.enemy, ["brightwing", "cassia", "dehaka", "valla", "artanis"])
        self.assertEqual(self.draft.map_slug, "sky-temple")
        await watcher.poll()
        self.assertEqual(self.said, ["Plan."])
        self.assertEqual(self.screen.reads[2:], [(FULL, True)])  # one quick look is enough

    async def test_what_the_draft_found_survives_into_the_loading_screen(self):
        watcher = self.briefed(shown("match_loading"))
        watcher.mode = OVER
        self.draft.add("banned", ["abathur"])
        await watcher.poll()
        self.assertEqual(self.draft.banned, ["abathur"])
        self.assertEqual(len(self.draft.enemy), 5)

    async def test_without_a_draft_the_loading_screen_is_the_whole_picture(self):
        self.draft.enemy.append("tychus")  # left over from an earlier match
        watcher = self.briefed(shown("match_loading"))
        await watcher.poll()
        self.assertNotIn("tychus", self.draft.enemy)

    async def test_nothing_is_said_without_a_briefing(self):
        watcher = self.briefed(shown("match_loading"), text=None)
        await watcher.poll()
        self.assertEqual(self.said, [])
        silent = self.briefed(shown("match_loading"), text="")
        await silent.poll()
        self.assertEqual(self.said, [])

    async def test_the_match_begins_when_the_loading_screen_is_gone(self):
        watcher = self.briefed(shown("match_loading"), shown("match_loading"), shown("loading"))
        await watcher.poll()
        for _ in range(LOST_AFTER - 1):
            await watcher.poll()
            self.assertEqual(watcher.mode, LOADING)
        await watcher.poll()
        self.assertEqual(watcher.mode, OVER)

    async def test_a_draft_screen_after_loading_starts_a_new_draft(self):
        watcher = self.briefed(shown("match_loading"), shown("ban_mine"))
        await watcher.poll()
        await watcher.poll()
        self.assertEqual(watcher.mode, DRAFT)
        self.assertEqual(self.draft.enemy, [])

    async def test_the_loading_screen_is_saved_to_the_screen_log(self):
        watcher = self.briefed(shown("match_loading"))
        await watcher.poll()
        await watcher.poll()
        saved = json.loads(next(self.folder.iterdir()).joinpath("draft.json").read_text())
        self.assertEqual([s["phase"] for s in saved["screens"]], ["loading"])

    async def test_a_read_on_request_notes_the_teams_without_briefing(self):
        watcher = self.briefed(shown("match_loading"))
        screen = await watcher.look()
        self.assertTrue(screen.is_loading)
        self.assertEqual(watcher.mode, LOADING)
        self.assertEqual(len(self.draft.enemy), 5)
        self.assertEqual(self.said, [])

    async def test_an_unreadable_screen_after_the_first_quick_look_is_a_problem(self):
        watcher = self.briefed(shown("match_loading"), ScreenError("blocked"))
        await watcher.poll()
        self.assertEqual(watcher.problem, "blocked")
        self.assertEqual(watcher.mode, IDLE)

    async def test_a_screen_that_stops_loading_between_the_two_reads_is_ignored(self):
        watcher = self.briefed(shown("match_loading"), shown("loading"))
        await watcher.poll()
        self.assertEqual(watcher.mode, IDLE)
        self.assertEqual(self.said, [])


class TestModes(WatcherCase):
    async def test_the_watcher_waits_with_quick_looks_until_a_draft_appears(self):
        watcher = self.watch(shown("loading"))
        await watcher.poll()
        self.assertEqual(watcher.mode, IDLE)
        self.assertEqual(self.screen.reads, [(FULL, True)])
        self.assertFalse(self.folder.exists())

    async def test_a_draft_screen_starts_a_draft_with_a_full_read(self):
        self.draft.enemy.append("dehaka")  # left over from the last draft
        watcher = self.watch(shown("ban_mine"))
        await watcher.poll()
        self.assertEqual(watcher.mode, DRAFT)
        self.assertEqual(self.screen.reads, [(FULL, True), (FULL, False)])
        self.assertEqual(self.draft.enemy, [])
        self.assertEqual(self.draft.map_slug, "sky-temple")
        self.assertIn("auriel", self.draft.ally)
        self.assertEqual(len(list(self.folder.iterdir())), 1)

    async def test_what_the_screen_shows_accumulates_and_never_unlocks(self):
        watcher = self.watch(
            shown("ban_mine"), shown("ban_mine"), shown("team_picking"), shown("ban_mine")
        )
        for _ in range(3):
            await watcher.poll()
        self.assertEqual(len(self.draft.enemy), 5)
        self.assertIn("brightwing", self.draft.enemy)

    async def test_a_hero_the_enemy_locked_leaves_my_team(self):
        watcher = self.watch(shown("ban_mine"), shown("ban_mine"), shown("ban_mine", ENEMY_ARTANIS))
        await watcher.poll()
        self.assertIn("artanis", self.draft.ally)
        await watcher.poll()
        self.assertNotIn("artanis", self.draft.ally)
        self.assertEqual(self.draft.enemy, ["artanis"])

    async def test_bans_join_the_draft(self):
        watcher = self.watch(shown("ban_mine"), shown("ban_mine"), shown("banned_announce"))
        await watcher.poll()
        await watcher.poll()
        self.assertEqual(self.draft.banned, ["abathur"])

    async def test_match_starting_ends_the_draft_and_the_watcher_goes_quiet(self):
        watcher = self.watch(
            shown("team_picking"), shown("team_picking"), shown("starting"), shown("loading")
        )
        await watcher.poll()
        await watcher.poll()
        self.assertEqual(watcher.mode, OVER)
        reads = len(self.screen.reads)
        await watcher.poll()
        self.assertEqual(watcher.mode, OVER)
        self.assertEqual(self.screen.reads[reads:], [(FULL, True)])  # only a quick look

    async def test_the_next_draft_forgets_the_old_one(self):
        watcher = self.watch(
            shown("team_picking"), shown("team_picking"), shown("starting"), shown("loading"),
            shown("ban_mine"), shown("ban_mine"),
        )  # fmt: skip
        for _ in range(3):
            await watcher.poll()
        self.assertEqual(len(self.draft.enemy), 5)
        self.log.clock = lambda: datetime(2026, 10, 30, 21, 0, 0)
        await watcher.poll()
        self.assertEqual(watcher.mode, DRAFT)
        self.assertEqual(self.draft.enemy, [])
        self.assertEqual(len(list(self.folder.iterdir())), 2)

    async def test_a_vanished_draft_screen_returns_to_waiting(self):
        watcher = self.watch(shown("ban_mine"), shown("ban_mine"), shown("loading"))
        await watcher.poll()
        for _ in range(LOST_AFTER - 1):
            await watcher.poll()
        self.assertEqual(watcher.mode, DRAFT)
        await watcher.poll()
        self.assertEqual(watcher.mode, IDLE)


class TestLog(WatcherCase):
    def saved(self) -> list[dict]:
        draft = next(self.folder.iterdir())
        return json.loads((draft / "draft.json").read_text())["screens"]

    async def test_a_screen_is_saved_only_when_what_it_shows_changed(self):
        watcher = self.watch(
            shown("ban_mine"), shown("ban_mine"), shown("ban_mine"), shown("team_picking")
        )
        for _ in range(3):
            await watcher.poll()
        screens = self.saved()
        self.assertEqual([s["file"] for s in screens], ["01.png", "02.png"])
        self.assertEqual(screens[0]["phase"], "ban")
        self.assertEqual(screens[1]["phase"], "pick")

    async def test_the_last_screen_before_the_match_is_saved_too(self):
        watcher = self.watch(shown("team_picking"), shown("team_picking"), shown("starting"))
        await watcher.poll()
        await watcher.poll()
        self.assertEqual([s["phase"] for s in self.saved()], ["pick", "starting"])

    async def test_without_a_log_nothing_is_written(self):
        watcher = self.watch(shown("ban_mine"), log=False)
        await watcher.poll()
        self.assertFalse(self.folder.exists())


class TestProblems(WatcherCase):
    async def test_an_unreadable_screen_is_reported_not_raised(self):
        watcher = self.watch(
            ScreenError("allow screen recording"), shown("ban_mine"), shown("ban_mine")
        )
        await watcher.poll()
        self.assertEqual(watcher.problem, "allow screen recording")
        self.assertEqual(watcher.mode, IDLE)
        await watcher.poll()
        self.assertIsNone(watcher.problem)
        self.assertEqual(watcher.mode, DRAFT)

    async def test_a_read_on_request_is_not_the_draft_in_progress_for_long(self):
        watcher = self.watch(shown("team_picking"), shown("ban_mine"))
        await watcher.look()
        self.assertEqual(watcher.mode, DRAFT)
        self.clock.now += STALE_S + 1
        screen = await watcher.look()
        self.assertEqual(screen.phase.kind, "ban")
        self.assertEqual(len(list(self.folder.iterdir())), 2)  # it started a new draft log

    async def test_a_window_somewhere_else_on_the_screen_is_read_the_same(self):
        layout = Layout((1600, 940), 900 / 1080, 760.0, 40.0)
        frames = [place(shown("our_pick"), layout), place(shown("our_pick"), layout)]
        watcher = self.watch(*frames)
        watcher.calibration = Calibration()
        await watcher.poll()
        self.assertEqual(self.draft.enemy, ["brightwing", "valla", "artanis"])
        self.assertEqual(watcher.screen.phase.kind, "pick")
        self.assertAlmostEqual(watcher.calibration.layout(frames[0], MAPS).scale, layout.scale, 2)

    async def test_the_loop_waits_between_looks_and_longer_after_a_problem(self):
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)
            if len(waits) == 3:
                raise asyncio.CancelledError

        watcher = self.watch(shown("loading"), ScreenError("blocked"), ScreenError("blocked"))
        watcher._sleep = sleep
        watcher.start()
        with self.assertRaises(asyncio.CancelledError):
            await watcher.run(self.say)
        self.assertEqual(waits, [watcher.poll_s, PROBLEM_RETRY_S, PROBLEM_RETRY_S])

    async def test_the_screen_is_not_looked_at_until_game_mode_starts(self):
        watcher = self.watch(shown("ban_mine"))
        task = asyncio.create_task(watcher.run(self.say))
        await asyncio.sleep(0.01)
        self.assertEqual(self.screen.reads, [])
        watcher.start()
        await asyncio.sleep(0.01)
        self.assertTrue(self.screen.reads)
        reads = len(self.screen.reads)
        watcher.stop()
        self.assertFalse(watcher.watching)
        await asyncio.sleep(0.01)
        self.assertEqual(len(self.screen.reads), reads)
        task.cancel()

    async def test_an_unexpected_failure_does_not_end_the_watching(self):
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)
            if len(waits) == 2:
                raise asyncio.CancelledError

        watcher = self.watch(shown("ban_mine"))
        watcher.reader = BrokenOnce(self.screen)
        watcher._sleep = sleep
        watcher.start()
        with self.assertLogs("voicemate.hots.screen.watcher", "ERROR"):
            with self.assertRaises(asyncio.CancelledError):
                await watcher.run(self.say)
        self.assertEqual(waits[0], PROBLEM_RETRY_S)  # the first look failed, the loop went on
        self.assertEqual(watcher.mode, DRAFT)  # the second look worked

    async def test_the_watcher_slows_down_when_no_draft_ever_shows(self):
        watcher = self.watch(shown("loading"))
        for _ in range(SLOW_AFTER):
            await watcher.poll()
        self.assertEqual(watcher._interval(), watcher.poll_s * SLOW_FACTOR)

    async def say(self, answer):
        self.said.append(answer.text)


class TestCues(WatcherCase):
    async def test_the_turn_is_announced_once_when_proactive(self):
        watcher = self.watch(
            shown("ban_mine"), shown("ban_mine"), shown("ban_mine"), shown("team_picking"),
            shown("our_pick"), proactive=True,
        )  # fmt: skip
        for _ in range(4):
            await watcher.poll()
        self.assertEqual(self.said, ["Your ban.", "Your pick."])

    async def test_nothing_is_said_by_default(self):
        watcher = self.watch(shown("ban_mine"))
        await watcher.poll()
        self.assertEqual(self.said, [])


class TestLook(WatcherCase):
    async def test_a_recent_read_is_reused(self):
        watcher = self.watch(shown("ban_mine"), shown("ban_mine"), shown("team_picking"))
        await watcher.poll()
        reads = len(self.screen.reads)
        self.clock.now = FRESH_S - 0.5
        first = await watcher.look()
        self.assertEqual(len(self.screen.reads), reads)
        self.assertEqual(first.phase.kind, "ban")
        self.clock.now = FRESH_S + 1
        second = await watcher.look()
        self.assertEqual(len(self.screen.reads), reads + 1)
        self.assertEqual(second.phase.kind, "pick")

    async def test_looking_at_a_draft_screen_starts_the_draft(self):
        watcher = self.watch(shown("team_picking"))
        screen = await watcher.look()
        self.assertTrue(screen.is_draft)
        self.assertEqual(watcher.mode, DRAFT)
        self.assertEqual(len(self.draft.enemy), 5)

    async def test_looking_at_something_else_changes_nothing(self):
        watcher = self.watch(shown("loading"))
        screen = await watcher.look()
        self.assertFalse(screen.is_draft)
        self.assertEqual(watcher.mode, IDLE)

    async def test_an_unreadable_screen_raises_for_the_caller(self):
        watcher = self.watch(ScreenError("blocked"))
        with self.assertRaises(ScreenError):
            await watcher.look()


if __name__ == "__main__":
    unittest.main()
