import unittest

import numpy as np

from tests.helpers import imported_modules, module_source
from tests.hots.screen.screens import boxes
from voicemate.hots.roster import Hero, Roster
from voicemate.hots.screen.interpreter import DraftScreen, Phase, interpret
from voicemate.hots.screen.reader import Frame, TextBox

HEROES = Roster(
    Hero(slug, name)
    for slug, name in [
        ("auriel", "Auriel"),
        ("qhira", "Qhira"),
        ("jaina", "Jaina"),
        ("johanna", "Johanna"),
        ("artanis", "Artanis"),
        ("abathur", "Abathur"),
        ("zagara", "Zagara"),
        ("tychus", "Tychus"),
        ("thrall", "Thrall"),
        ("brightwing", "Brightwing"),
        ("valla", "Valla"),
        ("cassia", "Cassia"),
        ("dehaka", "Dehaka"),
        ("xalatath", "Xal'atath"),
    ]
)
MAPS = {
    "sky-temple": "Sky Temple",
    "cursed-hollow": "Cursed Hollow",
    "hanamura-temple": "Hanamura Temple",
}


def read(name: str, *extra: TextBox) -> DraftScreen:
    frame = Frame(np.zeros((9, 16, 3), dtype=np.uint8), (*boxes(name), *extra))
    return interpret(frame, HEROES, MAPS)


class TestPhase(unittest.TestCase):
    def test_my_ban_turn(self):
        self.assertEqual(read("ban_mine").phase, Phase("ban", mine=True))

    def test_waiting_for_the_enemy_ban_is_not_my_turn(self):
        self.assertEqual(read("waiting_enemy_ban").phase, Phase("ban", mine=False))
        self.assertEqual(read("bans_first").phase, Phase("ban", mine=False))

    def test_picking_phases(self):
        self.assertEqual(read("enemy_picking").phase, Phase("pick", mine=False))
        self.assertEqual(read("team_picking").phase, Phase("pick", mine=False))
        self.assertEqual(read("our_pick").phase, Phase("pick", mine=True))

    def test_match_starting_ends_the_draft(self):
        screen = read("starting")
        self.assertEqual(screen.phase, Phase("starting"))
        self.assertFalse(screen.is_draft)

    def test_other_screens_are_not_a_draft(self):
        screen = read("loading")
        self.assertEqual(screen.phase, Phase())
        self.assertFalse(screen.is_draft)
        self.assertTrue(read("ban_mine").is_draft)


class TestMapAndTimer(unittest.TestCase):
    def test_map_name_comes_from_the_top_strip(self):
        self.assertEqual(read("ban_mine").map_slug, "sky-temple")
        self.assertIsNone(read("starting").map_slug)

    def test_a_map_word_elsewhere_is_ignored(self):
        extra = TextBox("CURSED HOLLOW", 1.0, 0.4, 0.5, 0.2, 0.03)
        self.assertEqual(read("ban_mine", extra).map_slug, "sky-temple")

    def test_timer_digits(self):
        self.assertEqual(read("ban_mine").seconds, 16)
        self.assertEqual(read("enemy_picking").seconds, 3)
        self.assertIsNone(read("banned_announce").seconds)


class TestTeams(unittest.TestCase):
    def test_hero_labels_are_split_by_side_and_ordered_top_down(self):
        screen = read("team_picking")
        self.assertEqual(screen.ally, ("auriel", "tychus", "jaina", "johanna", "thrall"))
        self.assertEqual(screen.enemy, ("brightwing", "cassia", "dehaka", "valla", "artanis"))

    def test_enemy_names_hide_until_they_lock(self):
        self.assertEqual(read("ban_mine").enemy, ())
        self.assertEqual(read("enemy_picking").enemy, ("brightwing", "artanis"))

    def test_player_names_and_animation_text_are_ignored(self):
        # "PLAYER" mid-screen is an announcement, "Player" at the edges is a mixed-case name.
        screen = read("enemy_picking")
        self.assertNotIn("player", screen.ally + screen.enemy)

    def test_a_hero_the_enemy_locked_is_invalid_for_me(self):
        enemy_artanis = TextBox("ARTANIS", 1.0, 0.95, 0.77, 0.045, 0.06)
        screen = read("ban_mine", enemy_artanis)
        self.assertEqual(screen.invalid, ("artanis",))
        self.assertNotIn("artanis", screen.ally)
        self.assertEqual(screen.enemy, ("artanis",))

    def test_unsure_text_is_not_a_hero(self):
        faint = TextBox("TYCHUS", 0.3, 0.005, 0.2, 0.04, 0.05)
        self.assertNotIn("tychus", read("ban_mine", faint).ally)

    def test_a_banned_announcement_names_the_hero_above_it(self):
        self.assertEqual(read("banned_announce").banned, ("abathur",))
        self.assertEqual(read("ban_mine").banned, ())

    def test_a_banned_hero_on_my_side_is_invalid(self):
        banner = (
            TextBox("JAINA", 1.0, 0.47, 0.50, 0.06, 0.04),
            TextBox("BANNED", 1.0, 0.48, 0.56, 0.05, 0.03),
        )
        screen = read("ban_mine", *banner)
        self.assertEqual(screen.banned, ("jaina",))
        self.assertIn("jaina", screen.invalid)


class TestPurity(unittest.TestCase):
    def test_the_interpreter_does_no_io_and_knows_no_macos(self):
        modules = imported_modules(module_source("hots/screen/interpreter.py"))
        for forbidden in ("Quartz", "ocrmac", "PIL", "pathlib", "subprocess", "httpx", "sqlite3"):
            self.assertNotIn(forbidden, modules)


if __name__ == "__main__":
    unittest.main()
