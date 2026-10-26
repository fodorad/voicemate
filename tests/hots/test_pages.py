import os
import unittest
from pathlib import Path

from tests.hots.builders import guide_html, text_html, tier_html
from voicemate.config import PROJECT_DIR
from voicemate.hots import pages


class TestParseGuide(unittest.TestCase):
    def setUp(self):
        self.guide = pages.parse_guide(
            guide_html(
                "li-ming",
                "Li-Ming",
                "Ranged Assassins",
                strengths=("Huge burst", "Great waveclear"),
                weaknesses=("Squishy",),
                synergies=("tyrael", "uther"),
                counters=("zeratul",),
                maps={
                    "cursed-hollow": "stronger",
                    "dragon-shire": "weaker",
                    "sky-temple": "average",
                },
            ),
            "li-ming",
        )

    def test_identity_and_role(self):
        self.assertEqual((self.guide.slug, self.guide.name), ("li-ming", "Li-Ming"))
        self.assertEqual(self.guide.role, "Ranged Assassin")
        self.assertEqual(self.guide.modified, "2026-01-01T00:00:00+00:00")

    def test_strengths_weaknesses_and_relations_keep_page_order(self):
        self.assertEqual(self.guide.strengths, ("Huge burst", "Great waveclear"))
        self.assertEqual(self.guide.weaknesses, ("Squishy",))
        self.assertEqual(self.guide.synergies, ("tyrael", "uther"))
        self.assertEqual(self.guide.counters, ("zeratul",))
        self.assertIn("likes friends", self.guide.synergy_text)
        self.assertIn("fears foes", self.guide.counter_text)

    def test_map_ratings(self):
        self.assertEqual(
            self.guide.maps,
            {"cursed-hollow": "stronger", "dragon-shire": "weaker", "sky-temple": "average"},
        )

    def test_repeated_portraits_count_once(self):
        guide = pages.parse_guide(guide_html("a", "A", synergies=("b", "b")), "a")
        self.assertEqual(guide.synergies, ("b",))

    def test_page_without_the_expected_blocks_is_an_error(self):
        with self.assertRaises(pages.PageError):
            pages.parse_guide("<html><body>nothing</body></html>", "x")
        no_block = guide_html("a", "A").replace("heroes_synergies_counters", "other")
        with self.assertRaises(pages.PageError):
            pages.parse_guide(no_block, "a")


class TestParseTierList(unittest.TestCase):
    def test_heroes_get_tier_role_and_ban_flag(self):
        html = tier_html(
            {
                ("S", "tank"): [("johanna", True)],
                ("A", "healer"): [("uther", False), ("rehgar", False)],
            }
        )
        entries = pages.parse_tier_list(html).entries
        self.assertEqual(
            [(e.hero, e.tier, e.role, e.ban) for e in entries],
            [
                ("johanna", "S", "tank", True),
                ("uther", "A", "healer", False),
                ("rehgar", "A", "healer", False),
            ],
        )

    def test_page_without_heroes_is_an_error(self):
        with self.assertRaises(pages.PageError):
            pages.parse_tier_list("<html><body><h2 id='tier-s'>Tier S</h2></body></html>")


class TestParseTextPage(unittest.TestCase):
    def test_sections_follow_the_headings(self):
        page = pages.parse_text_page(
            text_html(
                "Cursed Hollow Map Guide", {"Draft Strategy": "Pick  waveclear .", "Empty": ""}
            )
        )
        self.assertEqual(page.title, "Cursed Hollow Map Guide")
        self.assertEqual(page.sections, (("Draft Strategy", "Pick waveclear."),))

    def test_page_without_title_is_an_error(self):
        with self.assertRaises(pages.PageError):
            pages.parse_text_page("<html><body><h2>x</h2></body></html>")


REAL = PROJECT_DIR / "data" / "icyveins" / "raw"


@unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION") and REAL.exists(), "saved pages")
class TestRealPages(unittest.TestCase):  # pragma: no cover - needs the saved pages
    def test_every_saved_guide_parses_with_relations_and_maps(self):
        files = sorted(Path(REAL).glob("heroes_*-build-guide.html"))
        self.assertGreater(len(files), 50)
        for path in files:
            slug = path.name.removeprefix("heroes_").removesuffix("-build-guide.html")
            guide = pages.parse_guide(path.read_text(encoding="utf-8"), slug)
            with self.subTest(hero=slug):
                self.assertTrue(guide.counters and guide.synergies and guide.maps)


if __name__ == "__main__":
    unittest.main()
