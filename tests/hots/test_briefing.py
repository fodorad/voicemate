import tempfile
import unittest
from pathlib import Path

from tests.hots.builders import guide_html, write_pages
from voicemate.hots.advisor import Advisor
from voicemate.hots.briefing import EVEN, NO_TEAMS, NOTES, Briefing
from voicemate.hots.knowledge import Knowledge

PAGES = {
    "uther-build-guide": guide_html("uther", "Uther", "Healers", synergies=("muradin",)),
    "auriel-build-guide": guide_html("auriel", "Auriel", "Healers"),
    "abathur-build-guide": guide_html("abathur", "Abathur", "Supports"),
    "muradin-build-guide": guide_html(
        "muradin", "Muradin", "Tanks", counters=("zeratul",), maps={"cursed-hollow": "stronger"}
    ),
    "diablo-build-guide": guide_html("diablo", "Diablo", "Tanks", counters=("zeratul", "jaina")),
    "johanna-build-guide": guide_html("johanna", "Johanna", "Tanks"),
    "thrall-build-guide": guide_html("thrall", "Thrall", "Bruisers"),
    "zeratul-build-guide": guide_html(
        "zeratul", "Zeratul", "Melee Assassins", counters=("diablo",)
    ),
    "tychus-build-guide": guide_html(
        "tychus",
        "Tychus",
        "Ranged Assassins",
        counters=("valla",),
        maps={"cursed-hollow": "weaker"},
    ),
    "jaina-build-guide": guide_html("jaina", "Jaina", "Ranged Assassins", counters=("diablo",)),
    "valla-build-guide": guide_html(
        "valla", "Valla", "Ranged Assassins", maps={"cursed-hollow": "stronger"}
    ),
    "cassia-build-guide": guide_html("cassia", "Cassia", "Ranged Assassins"),
    "cursed-hollow-map-guide": (
        "<html><head><title>Cursed Hollow Map Guide - x</title></head></html>"
    ),
}


class BriefingCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write_pages(Path(tmp.name) / "pages", PAGES)
        knowledge = Knowledge(Path(tmp.name) / "hots.sqlite", Path(tmp.name) / "pages")
        knowledge.refresh()
        self.addCleanup(knowledge.close)
        self.briefing = Briefing(Advisor(knowledge))

    def texts(self, ally, enemy, map_slug=None) -> list[str]:
        return [note.text for note in self.briefing.notes(ally, enemy, map_slug)]


class TestShape(BriefingCase):
    def test_a_team_without_healer_or_frontline_is_told_so_first(self):
        said = self.briefing.say(["tychus", "jaina", "zeratul"], ["uther", "muradin"])
        self.assertTrue(
            said.startswith("No healer: don't overextend. No frontline: don't fight in the open.")
        )

    def test_a_support_hero_softens_the_missing_healer(self):
        notes = self.texts(["abathur", "muradin"], ["uther"])
        self.assertIn("No healer: play safe.", notes)
        self.assertNotIn("No healer: don't overextend.", notes)

    def test_a_healer_and_a_frontline_need_no_warning(self):
        notes = self.texts(["uther", "muradin"], ["uther", "muradin"])
        self.assertFalse([t for t in notes if t.startswith("No ")])

    def test_four_assassins_with_one_frontline_are_squishy(self):
        ally = ["tychus", "jaina", "zeratul", "valla", "diablo"]
        self.assertIn("Squishy team: stay grouped.", self.texts(ally, ["uther"]))
        self.assertNotIn("Squishy team: stay grouped.", self.texts([*ally[:3], "diablo"], []))

    def test_the_enemy_shape_is_noted(self):
        self.assertIn("No enemy healer: keep trading.", self.texts(["uther"], ["muradin"]))
        self.assertIn(
            "Two enemy healers: kill one first.", self.texts(["uther"], ["uther", "auriel"])
        )
        self.assertIn(
            "Enemy is squishy: dive them.", self.texts(["uther"], ["tychus", "jaina", "valla"])
        )


class TestMatchups(BriefingCase):
    def test_your_counters_of_an_enemy_hero_are_named(self):
        self.assertIn("Zeratul counters Muradin.", self.texts(["zeratul", "uther"], ["muradin"]))

    def test_the_foes_one_hero_counters_share_a_sentence(self):
        notes = self.texts(["zeratul", "uther"], ["muradin", "diablo", "valla"])
        self.assertIn("Zeratul counters Muradin and Diablo.", notes)

    def test_an_enemy_that_counters_your_hero_is_a_warning(self):
        self.assertIn("Valla counters Tychus: protect.", self.texts(["tychus", "uther"], ["valla"]))

    def test_the_heroes_one_enemy_counters_share_a_sentence(self):
        notes = self.texts(["zeratul", "jaina"], ["diablo"])
        self.assertIn("Diablo counters Zeratul and Jaina: protect.", notes)


class TestMapAndCombos(BriefingCase):
    def test_heroes_that_do_well_or_badly_on_the_map_are_named(self):
        notes = self.texts(["muradin", "tychus"], ["valla"], "cursed-hollow")
        self.assertIn("Strong on Cursed Hollow: Muradin.", notes)
        self.assertIn("Weak on Cursed Hollow: Tychus.", notes)
        self.assertIn("Enemy strong on Cursed Hollow: Valla.", notes)

    def test_no_map_no_map_notes(self):
        self.assertFalse([t for t in self.texts(["muradin"], ["valla"]) if "Hollow" in t])

    def test_two_heroes_that_work_together_are_a_combo(self):
        self.assertIn("Muradin and Uther combo well.", self.texts(["muradin", "uther"], []))
        self.assertIn("Uther and Muradin combo well.", self.texts(["uther", "muradin"], []))


class TestSay(BriefingCase):
    def test_only_the_heaviest_notes_are_said_heaviest_first(self):
        lineup = (["tychus", "jaina", "zeratul", "valla"], ["muradin", "valla"], "cursed-hollow")
        notes = sorted(self.briefing.notes(*lineup), key=lambda note: -note.weight)
        self.assertGreater(len(notes), NOTES)
        self.assertEqual(self.briefing.say(*lineup), " ".join(n.text for n in notes[:NOTES]))
        self.assertTrue(self.briefing.say(*lineup).startswith("No healer: don't overextend."))

    def test_a_balanced_lineup_has_no_edge(self):
        self.assertEqual(self.briefing.say(["uther", "johanna"], ["auriel", "diablo"]), EVEN)

    def test_without_heroes_there_is_nothing_to_say(self):
        self.assertEqual(self.briefing.say([], []), NO_TEAMS)

    def test_unknown_heroes_are_skipped(self):
        self.assertEqual(self.briefing.say(["nobody"], ["ghost"]), NO_TEAMS)


if __name__ == "__main__":
    unittest.main()
