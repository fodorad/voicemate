import os
import tempfile
import unittest
from pathlib import Path

from tests.hots.builders import guide_html, text_html, tier_html, write_pages
from voicemate.config import PROJECT_DIR
from voicemate.hots.knowledge import Knowledge, PageId, TierRank, classify

PAGES = {
    "muradin-build-guide": guide_html(
        "muradin",
        "Muradin",
        "Tanks",
        synergies=("uther",),
        counters=("zeratul", "tychus"),
        maps={"cursed-hollow": "stronger", "dragon-shire": "weaker"},
    ),
    "zeratul-build-guide": guide_html(
        "zeratul", "Zeratul", "Melee Assassins", counters=("muradin",), synergies=("uther",)
    ),
    "heroes-of-the-storm-general-tier-list": tier_html(
        {("S", "tank"): [("muradin", True)], ("B", "assassin"): [("zeratul", False)]}
    ),
    "heroes-of-the-storm-cursed-hollow-tier-list": tier_html(
        {("A", "tank"): [("muradin", False)], ("S", "assassin"): [("ghost", False)]}
    ),
    "cursed-hollow-map-guide": text_html(
        "Cursed Hollow Map Guide", {"Draft Strategy": "Bring waveclear.", "Tributes": "Take them."}
    ),
    "muradin-talents": text_html("Muradin's Talents", {"Level 1": "Pick Bolt."}),
}


class KnowledgeCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pages = Path(tmp.name) / "pages"
        write_pages(self.pages, PAGES)
        self.database = Path(tmp.name) / "hots.sqlite"
        self.kb = Knowledge(self.database, self.pages)
        self.addCleanup(self.kb.close)

    def page_file(self, stem: str) -> Path:
        return self.pages / f"heroes_{stem}.html"


class TestClassify(unittest.TestCase):
    def test_file_stems_map_to_page_kinds(self):
        self.assertEqual(classify("muradin-build-guide"), PageId("guide", "muradin"))
        self.assertEqual(classify("muradin-talents"), PageId("talents", "muradin"))
        self.assertEqual(
            classify("muradin-abilities-strategy"), PageId("abilities-strategy", "muradin")
        )
        self.assertEqual(classify("dragon-shire-map-guide"), PageId("map", "dragon-shire"))
        self.assertEqual(
            classify("heroes-of-the-storm-dragon-shire-tier-list"), PageId("tiers", "dragon-shire")
        )

    def test_other_pages_are_ignored(self):
        for stem in ("changelog", "heroes-of-the-storm-tier-list-hub", "x-map-strategy-guide"):
            self.assertIsNone(classify(stem), stem)


class TestRefresh(KnowledgeCase):
    def test_first_refresh_imports_everything_and_the_second_nothing(self):
        first = self.kb.refresh()
        self.assertEqual(len(first.added), len(PAGES))
        self.assertTrue(first.changed)
        second = self.kb.refresh()
        self.assertFalse(second.changed)
        self.assertEqual(second.unchanged, len(PAGES))
        self.assertEqual(second.summary(), "")

    def test_only_a_changed_page_is_imported_again(self):
        self.kb.refresh()
        write_pages(
            self.pages,
            {"muradin-build-guide": guide_html("muradin", "Muradin", counters=("tychus",))},
        )
        report = self.kb.refresh()
        self.assertEqual(report.updated, ("muradin-build-guide",))
        self.assertEqual(report.added, ())
        self.assertEqual(self.kb.counters_of("muradin"), ("tychus",))
        self.assertIn("1 updated (muradin-build-guide)", report.summary())

    def test_a_deleted_page_loses_its_rows(self):
        self.kb.refresh()
        self.page_file("zeratul-build-guide").unlink()
        report = self.kb.refresh()
        self.assertEqual(report.removed, ("zeratul-build-guide",))
        self.assertIsNone(self.kb.roster().get("zeratul"))
        self.assertEqual(self.kb.counters_of("zeratul"), ())
        self.assertIn("1 removed", report.summary())

    def test_an_unreadable_page_is_reported_and_keeps_its_old_rows(self):
        self.kb.refresh()
        self.page_file("muradin-build-guide").write_text("<html>broken</html>")
        with self.assertLogs("voicemate.hots.knowledge", "WARNING"):
            report = self.kb.refresh()
        self.assertEqual([page for page, _ in report.failed], ["muradin-build-guide"])
        self.assertEqual(self.kb.counters_of("muradin"), ("zeratul", "tychus"))
        self.assertIn("unreadable", report.summary())

    def test_missing_folder_and_unknown_files_are_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Knowledge(Path(tmp) / "a.sqlite", Path(tmp) / "nowhere")
            self.assertFalse(empty.refresh().changed)
            empty.close()
        (self.pages / "heroes_changelog.html").write_text("<html></html>")
        self.assertEqual(len(self.kb.refresh().added), len(PAGES))

    def test_many_added_pages_are_summarised_briefly(self):
        summary = self.kb.refresh().summary()
        self.assertTrue(summary.startswith("HotS pages: 6 new ("))
        self.assertIn("…", summary)

    def test_the_database_survives_a_restart(self):
        self.kb.refresh()
        reopened = Knowledge(
            Path(self.kb._db.execute("PRAGMA database_list").fetchone()[2]), self.pages
        )
        self.addCleanup(reopened.close)
        self.assertFalse(reopened.refresh().changed)
        self.assertEqual(reopened.synergies_of("muradin"), ("uther",))


class TestQueries(KnowledgeCase):
    def setUp(self):
        super().setUp()
        self.kb.refresh()

    def test_roster_has_the_guides_heroes(self):
        roster = self.kb.roster()
        self.assertEqual(sorted(h.slug for h in roster), ["muradin", "zeratul"])
        self.assertEqual(roster.get("zeratul").role, "Melee Assassin")

    def test_relations_keep_page_order_and_work_both_ways(self):
        self.assertEqual(self.kb.counters_of("muradin"), ("zeratul", "tychus"))
        self.assertEqual(self.kb.synergies_of("zeratul"), ("uther",))
        self.assertEqual(self.kb.countered_by("muradin"), ("zeratul",))
        self.assertEqual(self.kb.counters_of("nobody"), ())

    def test_map_ratings_and_names(self):
        self.assertEqual(self.kb.map_rating("muradin", "cursed-hollow"), "stronger")
        self.assertEqual(self.kb.map_rating("muradin", "dragon-shire"), "weaker")
        self.assertIsNone(self.kb.map_rating("muradin", "sky-temple"))
        self.assertEqual(self.kb.maps(), {"cursed-hollow": "Cursed Hollow"})

    def test_tiers_per_list(self):
        self.assertEqual(self.kb.tier("muradin", "general"), TierRank("S", "tank", True))
        self.assertEqual(self.kb.tier("muradin", "cursed-hollow"), TierRank("A", "tank", False))
        self.assertIsNone(self.kb.tier("muradin", "aram"))

    def test_text_sections_are_kept_for_follow_ups(self):
        texts = {(page, heading): text for page, heading, text in self.kb.chunks()}
        self.assertEqual(texts[("cursed-hollow-map-guide", "Draft Strategy")], "Bring waveclear.")
        self.assertIn(("muradin-build-guide", "Counters"), texts)
        self.assertEqual(len(self.kb.chunks("map")), 2)


REAL = PROJECT_DIR / "data" / "icyveins" / "raw"


@unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION") and REAL.exists(), "saved pages")
class TestRealPages(unittest.TestCase):  # pragma: no cover - needs the saved pages
    def test_all_saved_pages_import_and_the_second_start_is_instant(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            kb = Knowledge(Path(tmp) / "hots.sqlite", REAL)
            first = kb.refresh()
            self.assertEqual(first.failed, ())
            self.assertGreaterEqual(len(kb.roster()), 90)
            started = time.perf_counter()
            self.assertFalse(kb.refresh().changed)
            self.assertLess(time.perf_counter() - started, 1.0)
            self.assertTrue(kb.counters_of("zuljin"))
            kb.close()


if __name__ == "__main__":
    unittest.main()
