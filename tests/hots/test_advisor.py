import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import imported_modules, module_source, uses_async_syntax
from tests.hots.builders import guide_html, tier_html, write_pages
from voicemate.hots.advisor import Advisor
from voicemate.hots.knowledge import Knowledge

PAGES = {
    "muradin-build-guide": guide_html(
        "muradin",
        "Muradin",
        counters=("zeratul", "tychus"),
        synergies=("uther",),
        maps={"cursed-hollow": "stronger"},
    ),
    "zeratul-build-guide": guide_html(
        "zeratul",
        "Zeratul",
        "Melee Assassins",
        counters=("muradin",),
        synergies=("uther", "jaina"),
        maps={"cursed-hollow": "weaker"},
    ),
    "tychus-build-guide": guide_html(
        "tychus",
        "Tychus",
        "Ranged Assassins",
        counters=("diablo",),
        maps={"cursed-hollow": "average"},
    ),
    "uther-build-guide": guide_html("uther", "Uther", "Healers", synergies=("muradin", "zeratul")),
    "jaina-build-guide": guide_html("jaina", "Jaina", "Ranged Assassins", counters=("zeratul",)),
    "diablo-build-guide": guide_html("diablo", "Diablo"),
    "heroes-of-the-storm-general-tier-list": tier_html(
        {
            ("S", "tank"): [("muradin", True)],
            ("A", "ranged"): [("tychus", False)],
            ("B", "assassin"): [("zeratul", False)],
        }
    ),
    "heroes-of-the-storm-cursed-hollow-tier-list": tier_html(
        {
            ("A", "tank"): [("muradin", False)],
            ("S", "mage"): [("jaina", False)],
            ("D", "assassin"): [("zeratul", False)],
        }
    ),
    "cursed-hollow-map-guide": "<html><head><title>Cursed Hollow Map Guide - x</title></head>"
    "<body><div class='heading_container'><h2>Intro</h2></div><p>Text.</p></body></html>",
}


class AdvisorCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write_pages(Path(tmp.name) / "pages", PAGES)
        kb = Knowledge(Path(tmp.name) / "hots.sqlite", Path(tmp.name) / "pages")
        self.addCleanup(kb.close)
        kb.refresh()
        self.advisor = Advisor(kb)

    def names(self, suggestions):
        return [s.hero.slug for s in suggestions]


class TestCounters(AdvisorCase):
    def test_counters_are_ranked_by_tier_and_explain_themselves(self):
        result = self.advisor.counters("muradin")
        self.assertEqual(self.names(result), ["tychus", "zeratul"])
        self.assertEqual(result[0].score, 2.5)
        self.assertEqual(result[0].reasons, ("counters Muradin", "A tier overall"))

    def test_the_map_changes_the_ranking(self):
        result = self.advisor.counters("muradin", "cursed-hollow")
        self.assertEqual(self.names(result), ["tychus"])  # zeratul scores 0 there: D tier, weaker

    def test_unknown_hero_or_hero_without_counters_gives_nothing(self):
        self.assertEqual(self.advisor.counters("nobody"), [])
        self.assertEqual(self.advisor.counters("diablo"), [])


class TestPicks(AdvisorCase):
    def test_picks_counter_the_enemy_and_avoid_being_countered(self):
        result = self.advisor.picks(enemy=["muradin"])
        self.assertEqual(self.names(result), ["tychus", "zeratul"])
        self.assertEqual(result[1].score, 1.0)  # counters Muradin (+2), is countered back (-1)

    def test_allies_add_synergy(self):
        result = self.advisor.picks(enemy=["muradin"], ally=["uther"])
        self.assertEqual(self.names(result)[:2], ["tychus", "zeratul"])
        zeratul = next(s for s in result if s.hero.slug == "zeratul")
        self.assertEqual(zeratul.score, 1.5)
        self.assertIn("works with Uther", zeratul.reasons)

    def test_map_rating_and_tier_count_without_an_enemy(self):
        result = self.advisor.picks(enemy=[], map_slug="cursed-hollow")
        self.assertEqual(self.names(result), ["muradin", "jaina", "tychus"])
        self.assertEqual(result[0].reasons, ("stronger on Cursed Hollow", "A tier Cursed Hollow"))
        self.assertEqual([s.score for s in result], [1.5, 1.0, 0.5])
        self.assertNotIn("zeratul", self.names(result))  # weaker there and D tier

    def test_picked_and_banned_heroes_never_appear(self):
        result = self.advisor.picks(enemy=["muradin"], ally=["tychus"], banned=["zeratul"])
        self.assertTrue({"muradin", "tychus", "zeratul"}.isdisjoint(self.names(result)))

    def test_empty_draft_falls_back_to_the_general_tiers(self):
        self.assertEqual(self.names(self.advisor.picks(enemy=[]))[0], "muradin")


class TestBans(AdvisorCase):
    def test_bans_follow_the_enemy_synergy_and_the_ban_flag(self):
        result = self.advisor.bans(enemy=["uther"])
        self.assertEqual(self.names(result), ["muradin", "zeratul", "tychus"])
        self.assertEqual(result[0].score, 3.0)  # synergy 1 + S tier 1 + ban flag 1
        self.assertEqual(
            result[0].reasons, ("works with Uther", "S tier overall", "recommended ban")
        )

    def test_heroes_that_counter_my_allies_are_ban_candidates(self):
        result = self.advisor.bans(enemy=[], ally=["muradin"])
        self.assertIn(
            "counters your Muradin", next(s for s in result if s.hero.slug == "zeratul").reasons
        )

    def test_map_data_decides_the_bans_on_that_map(self):
        names = self.names(self.advisor.bans(enemy=[], map_slug="cursed-hollow"))
        self.assertEqual(names, ["muradin", "jaina", "tychus"])
        self.assertEqual(
            self.names(self.advisor.bans(enemy=[], map_slug="cursed-hollow", limit=1)), ["muradin"]
        )


class TestPurity(unittest.TestCase):
    def test_the_advisor_does_no_io_of_its_own(self):
        source = module_source("hots/advisor.py")
        self.assertFalse(uses_async_syntax(source))
        io = {"sqlite3", "httpx", "socket", "subprocess", "pathlib", "os", "time"}
        self.assertEqual({m.split(".")[0] for m in imported_modules(source)} & io, set())


class TestSpeed(unittest.TestCase):
    def test_a_full_roster_answers_well_inside_the_budget(self):
        pages = {}
        slugs = [f"hero{i}" for i in range(91)]
        for i, slug in enumerate(slugs):
            pages[f"{slug}-build-guide"] = guide_html(
                slug,
                slug.title(),
                counters=tuple(slugs[(i + k) % 91] for k in range(1, 6)),
                synergies=tuple(slugs[(i + k) % 91] for k in range(7, 12)),
                maps={"cursed-hollow": "stronger"},
            )
        with tempfile.TemporaryDirectory() as tmp:
            write_pages(Path(tmp) / "pages", pages)
            kb = Knowledge(Path(tmp) / "k.sqlite", Path(tmp) / "pages")
            kb.refresh()
            advisor = Advisor(kb)
            started = time.perf_counter()
            advisor.picks(enemy=slugs[:5], ally=slugs[5:9], map_slug="cursed-hollow")
            advisor.bans(enemy=slugs[:5], map_slug="cursed-hollow")
            self.assertLess(time.perf_counter() - started, 0.2)
            kb.close()


if __name__ == "__main__":
    unittest.main()
