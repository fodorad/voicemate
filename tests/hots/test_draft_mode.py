import tempfile
import unittest
from pathlib import Path

from tests.helpers import ScriptedScreen
from tests.hots.builders import guide_html, write_pages
from tests.hots.screen.screens import boxes
from tests.hots.screen.test_interpreter import screen_of
from voicemate.config import Config, HotsConfig, ScreenConfig
from voicemate.hots.commands import Command, parse
from voicemate.hots.intents import clean
from voicemate.hots.screen.reader import ScreenError
from voicemate.hots.screen.watcher import DRAFT, OVER
from voicemate.hots.skill import BLIND, SAY_AGAIN, Hots, ScreenKit

HEROES = {
    "auriel": ("Auriel", "Healers"),
    "qhira": ("Qhira", "Melee Assassins"),
    "jaina": ("Jaina", "Ranged Assassins"),
    "johanna": ("Johanna", "Tanks"),
    "artanis": ("Artanis", "Melee Assassins"),
    "abathur": ("Abathur", "Specialists"),
    "zagara": ("Zagara", "Specialists"),
    "tychus": ("Tychus", "Ranged Assassins"),
    "thrall": ("Thrall", "Melee Assassins"),
    "brightwing": ("Brightwing", "Healers"),
    "valla": ("Valla", "Ranged Assassins"),
    "cassia": ("Cassia", "Ranged Assassins"),
    "dehaka": ("Dehaka", "Melee Assassins"),
}
PAGES = {
    f"{slug}-build-guide": guide_html(
        slug, name, role, counters=("tychus",) if slug == "johanna" else ()
    )
    for slug, (name, role) in HEROES.items()
}
PAGES["sky-temple-map-guide"] = "<html><head><title>Sky Temple Map Guide - x</title></head></html>"


def shown(name: str):
    return screen_of(*boxes(name))


class TestParse(unittest.TestCase):
    def test_commands_about_the_screen_and_the_mode(self):
        for said, expected in [
            ("read the draft", Command("read")),
            ("ava please scan the screen", Command("read")),
            ("what do you see", Command("read")),
            ("read the draft and ban", Command("read", "ban")),
            ("read the last screenshot", Command("last")),
            ("draft mode", Command("on")),
            ("start draft mode", Command("on")),
            ("draft mode off", Command("off")),
            ("end draft", Command("off")),
        ]:
            with self.subTest(said=said):
                self.assertEqual(parse(clean(said)), expected)

    def test_ordinary_speech_is_not_a_command(self):
        for said in ("read me the news", "what is good against muradin", "draft", "check the time"):
            with self.subTest(said=said):
                self.assertIsNone(parse(clean(said)))


class DraftModeCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write_pages(Path(tmp.name) / "pages", PAGES)
        config = Config(
            data_dir=f"{tmp.name}/data",
            hots=HotsConfig(pages_dir=f"{tmp.name}/pages", screen=ScreenConfig(enabled=False)),
        )
        self.hots = Hots.open(config)
        self.addCleanup(self.hots.close)
        self.tmp = tmp.name

    def skill(self, *frames, saved=None):
        self.live = ScriptedScreen(*frames)
        self.hots.screen = ScreenKit(
            self.live, None, ScreenConfig(log_dir=f"{self.tmp}/screens"), saved
        )
        self.skill_ = self.hots.skill()
        return self.skill_

    async def ask(self, text: str) -> str | None:
        answer = await self.skill_.answer(text)
        return None if answer is None else answer.text


class TestDraftMode(DraftModeCase):
    async def test_outside_a_draft_ordinary_speech_goes_to_the_chat_model(self):
        self.skill(shown("loading"))
        self.assertFalse(self.skill_.draft_mode)
        self.assertIsNone(await self.ask("what time is it"))

    async def test_a_draft_on_screen_without_draft_mode_stays_ordinary(self):
        skill = self.skill(shown("ban_mine"))
        await skill.watcher.poll()
        self.assertFalse(skill.draft_mode)
        self.assertIsNone(await self.ask("what time is it"))

    async def test_draft_mode_gives_short_focused_answers(self):
        skill = self.skill(shown("ban_mine"))
        await self.ask("game mode on")
        await skill.watcher.poll()
        self.assertTrue(skill.draft_mode)
        self.assertEqual(await self.ask("what time is it"), SAY_AGAIN)
        self.assertEqual(await self.ask("what is good against johanna"), "Counter Johanna: Tychus.")

    async def test_why_gives_the_reason_of_the_last_advice(self):
        self.skill(shown("loading"))
        self.assertEqual(await self.ask("why"), "No advice yet.")
        await self.ask("what is good against johanna")
        self.assertEqual(await self.ask("why"), "Tychus: counters Johanna.")

    async def test_the_match_starting_ends_draft_mode(self):
        skill = self.skill(shown("team_picking"), shown("team_picking"), shown("starting"))
        await self.ask("draft mode on")
        await skill.watcher.poll()
        await skill.watcher.poll()
        self.assertEqual(skill.watcher.mode, OVER)
        self.assertFalse(skill.draft_mode)
        self.assertIsNone(await self.ask("what time is it"))

    async def test_draft_mode_can_be_switched_by_voice(self):
        self.skill(shown("loading"))
        self.assertEqual(await self.ask("draft mode on"), "Draft mode on.")
        self.assertTrue(self.skill_.draft_mode)
        self.assertEqual(await self.ask("what time is it"), SAY_AGAIN)
        self.assertEqual(await self.ask("end draft"), "Draft mode off.")
        self.assertFalse(self.skill_.draft_mode)
        self.assertIsNone(await self.ask("what time is it"))
        self.assertEqual(self.skill_.watcher.mode, OVER)

    async def test_draft_mode_pauses_for_the_match_and_resumes_with_the_next_draft(self):
        skill = self.skill(shown("starting"), shown("ban_mine"), shown("ban_mine"))
        await self.ask("draft mode on")
        skill.watcher.mode = DRAFT
        await skill.watcher.poll()
        self.assertFalse(skill.draft_mode)
        await skill.watcher.poll()
        self.assertTrue(skill.draft_mode)

    async def test_a_session_can_start_in_draft_mode(self):
        self.skill(shown("loading"))
        self.hots.screen.settings.game_mode = True
        skill = self.hots.skill()
        self.assertTrue(skill.draft_mode)
        self.assertTrue(skill.watcher.watching)


class TestReadCommands(DraftModeCase):
    async def test_read_the_draft_tells_map_teams_and_bans(self):
        self.skill(shown("banned_announce"))
        said = await self.ask("read the draft")
        self.assertTrue(said.startswith("Map Sky Temple. Team Auriel"), said)
        self.assertIn("Bans Abathur.", said)

    async def test_my_turn_adds_the_matching_advice(self):
        self.skill(shown("our_pick"))
        said = await self.ask("read the draft")
        self.assertIn("Enemy Brightwing, Valla, Artanis.", said)
        self.assertIn("pick", said.split("Artanis.")[1].lower())

    async def test_no_draft_on_screen_says_so(self):
        self.skill(shown("loading"))
        self.assertEqual(await self.ask("read the draft"), "I see no draft.")

    async def test_an_unreadable_screen_is_explained(self):
        self.skill(ScreenError("blocked"))
        self.assertEqual(await self.ask("read the draft"), BLIND)

    async def test_the_last_screenshot_is_read_instead_of_the_live_screen(self):
        saved = ScriptedScreen(shown("team_picking"))
        self.skill(shown("loading"), saved=saved)
        said = await self.ask("read the last screenshot")
        self.assertIn("Enemy Brightwing, Cassia, Dehaka, Valla, Artanis.", said)
        self.assertEqual(self.live.reads, [])
        self.assertEqual(len(saved.reads), 1)

    async def test_a_missing_screenshot_is_reported(self):
        self.skill(
            shown("loading"),
            saved=ScriptedScreen(ScreenError("There is no saved screenshot to read.")),
        )
        self.assertEqual(
            await self.ask("read the last screenshot"), "There is no saved screenshot to read."
        )

    async def test_without_screen_reading_the_command_is_refused(self):
        self.hots.screen = None
        self.skill_ = self.hots.skill()
        self.assertEqual(await self.ask("read the draft"), "I can't read the screen here.")


if __name__ == "__main__":
    unittest.main()
