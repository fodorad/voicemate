import tempfile
import unittest
from pathlib import Path

from tests.hots.builders import guide_html, write_pages
from voicemate.config import Config, HotsConfig
from voicemate.hots.aliases import AliasStore
from voicemate.hots.skill import Hots
from voicemate.pipeline.skills import SkillAnswer

PAGES = {
    "muradin-build-guide": guide_html("muradin", "Muradin", counters=("tychus",)),
    "tychus-build-guide": guide_html("tychus", "Tychus", "Ranged Assassins"),
}


class TestHots(unittest.TestCase):
    def config(self, tmp: str, **hots) -> Config:
        return Config(data_dir=f"{tmp}/data", hots=HotsConfig(pages_dir=f"{tmp}/pages", **hots))

    def test_pages_become_a_skill_that_answers_in_english(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pages(Path(tmp) / "pages", PAGES)
            hots = Hots.open(self.config(tmp))
            self.assertIsNotNone(hots)
            self.assertEqual(len(hots.report.added), 2)
            answer = hots.skill().answer("what is good against muradin")
            self.assertEqual(answer.lang, "en")
            self.assertTrue(answer.text.startswith("To counter Muradin: Tychus"))
            hots.close()

    def test_learned_spellings_are_known_at_the_next_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pages(Path(tmp) / "pages", PAGES)
            config = self.config(tmp, aliases_file=f"{tmp}/aliases.toml")
            AliasStore(config.hots.aliases_path).add("tichus", "tychus")
            hots = Hots.open(config)
            self.assertEqual(hots.skill().answer("enemy has tichus").text, "Enemy team: Tychus.")
            hots.close()

    def test_every_session_gets_its_own_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pages(Path(tmp) / "pages", PAGES)
            hots = Hots.open(self.config(tmp))
            first, second = hots.session_skills()[0], hots.session_skills()[0]
            first.answer("enemy has muradin")
            self.assertEqual(
                second.answer("enemy has tychus"), SkillAnswer("Enemy team: Tychus.", "en")
            )
            hots.close()

    def test_ordinary_questions_are_not_answered(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pages(Path(tmp) / "pages", PAGES)
            hots = Hots.open(self.config(tmp))
            self.assertIsNone(hots.skill().answer("what time is it"))
            hots.close()

    def test_second_start_reports_nothing_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pages(Path(tmp) / "pages", PAGES)
            Hots.open(self.config(tmp)).close()
            second = Hots.open(self.config(tmp))
            self.assertEqual(second.report.summary(), "")
            second.close()

    def test_disabled_or_no_saved_pages_means_no_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(Hots.open(self.config(tmp)))  # no pages folder yet
            write_pages(Path(tmp) / "pages", PAGES)
            self.assertIsNone(Hots.open(self.config(tmp, enabled=False)))


if __name__ == "__main__":
    unittest.main()
