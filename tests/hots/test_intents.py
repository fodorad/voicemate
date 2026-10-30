import tempfile
import unittest
from pathlib import Path

from tests.helpers import imported_modules, module_source
from tests.hots.builders import guide_html, tier_html, write_pages
from voicemate.hots.advisor import Advisor
from voicemate.hots.aliases import AliasStore
from voicemate.hots.draft import Draft
from voicemate.hots.intents import Intents
from voicemate.hots.knowledge import Knowledge


def _map_page(name: str) -> str:
    return f"<html><head><title>{name} Map Guide - x</title></head></html>"


PAGES = {
    "muradin-build-guide": guide_html(
        "muradin",
        "Muradin",
        counters=("zeratul", "tychus"),
        synergies=("uther",),
        maps={"cursed-hollow": "stronger"},
    ),
    "zeratul-build-guide": guide_html(
        "zeratul", "Zeratul", "Melee Assassins", counters=("muradin",), synergies=("uther",)
    ),
    "tychus-build-guide": guide_html("tychus", "Tychus", "Ranged Assassins", counters=("diablo",)),
    "tyrael-build-guide": guide_html("tyrael", "Tyrael", "Tanks"),
    "tyrande-build-guide": guide_html("tyrande", "Tyrande", "Healers"),
    "uther-build-guide": guide_html("uther", "Uther", "Healers", synergies=("muradin", "zeratul")),
    "diablo-build-guide": guide_html("diablo", "Diablo"),
    "li-li-build-guide": guide_html("li-li", "Li Li", "Healers"),
    "heroes-of-the-storm-general-tier-list": tier_html(
        {("S", "tank"): [("muradin", True)], ("A", "ranged"): [("tychus", False)]}
    ),
    "cursed-hollow-map-guide": _map_page("Cursed Hollow"),
    "dragon-shire-map-guide": _map_page("Dragon Shire"),
    "hanamura-temple-map-guide": _map_page("Hanamura Temple"),
    "sky-temple-map-guide": _map_page("Sky Temple"),
}


class IntentCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write_pages(Path(tmp.name) / "pages", PAGES)
        kb = Knowledge(Path(tmp.name) / "hots.sqlite", Path(tmp.name) / "pages")
        self.addCleanup(kb.close)
        kb.refresh()
        self.draft = Draft()
        self.intents = Intents(Advisor(kb), self.draft)

    def ask(self, text):
        return self.intents.handle(text)


class TestCounters(IntentCase):
    def test_counter_questions_name_the_best_heroes_and_a_reason(self):
        for text in (
            "What's good against Muradin?",
            "who counters muradin",
            "Ava, how do I counter Muradin",
            "best counterpick vs muradin",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.ask(text),
                    "To counter Muradin: Tychus, Zeratul. "
                    "Tychus: counters Muradin and A tier overall.",
                )

    def test_a_bare_counterpick_question_uses_the_draft(self):
        self.ask("enemy has muradin")
        self.assertTrue(self.ask("what is a good counterpick").startswith("Pick: Tychus"))

    def test_several_enemies_become_a_pick_question(self):
        answer = self.ask("what is good against tychus and muradin")
        self.assertTrue(answer.startswith("Pick: "))

    def test_hero_without_counters_says_so(self):
        self.assertEqual(self.ask("what counters diablo"), "I have no counters listed for Diablo.")

    def test_misheard_names_resolve_and_ambiguous_ones_ask(self):
        self.assertTrue(self.ask("who counters muradan").startswith("To counter Muradin"))
        self.assertEqual(self.ask("who counters tyra"), "Tyrael or Tyrande?")

    def test_names_nobody_knows_fall_through_to_the_chat_model(self):
        self.assertIsNone(self.ask("what is good against the cold weather"))


class TestClarify(IntentCase):
    def test_the_answer_to_a_question_runs_the_original_command(self):
        self.assertEqual(self.ask("who counters tyra"), "Tyrael or Tyrande?")
        self.assertEqual(self.ask("second"), "I have no counters listed for Tyrande.")

    def test_a_hero_name_answers_the_question(self):
        self.ask("enemy has tyra")
        self.assertEqual(self.ask("tyrael"), "Enemy team: Tyrael.")

    def test_yes_takes_the_first_candidate(self):
        self.ask("enemy has tyra")
        self.assertEqual(self.ask("yes"), "Enemy team: Tyrael.")

    def test_other_speech_drops_the_question(self):
        self.ask("who counters tyra")
        self.assertIsNone(self.ask("what time is it"))
        self.assertIsNone(self.ask("first"))

    def test_a_settled_spelling_is_learned_and_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AliasStore(Path(tmp) / "aliases.toml")
            self.intents.aliases = store
            self.ask("who counters tyra")
            self.ask("second")
            self.assertEqual(store.load(), {"tyra": "tyrande"})
            self.assertEqual(self.intents.advisor.roster.resolve("tyra").hero.slug, "tyrande")


class TestDraftCommands(IntentCase):
    def test_enemy_and_ally_picks_accumulate(self):
        self.assertEqual(self.ask("enemy has Tychus and Muradin"), "Enemy team: Tychus, Muradin.")
        self.assertEqual(self.ask("they also picked uther"), "Enemy team: Tychus, Muradin, Uther.")
        self.assertEqual(self.ask("my team has zeratul"), "Your team: Zeratul.")
        self.assertEqual(
            self.ask("add diablo to the enemy"), "Enemy team: Tychus, Muradin, Uther, Diablo."
        )
        self.assertEqual(
            (self.draft.enemy[:2], self.draft.ally), (["tychus", "muradin"], ["zeratul"])
        )

    def test_a_name_that_sounds_like_two_heroes_goes_to_the_one_still_available(self):
        self.ask("we have tyrael")
        self.assertEqual(self.ask("enemy has tyrand"), "Enemy team: Tyrande.")
        self.assertEqual(self.draft.ally, ["tyrael"])

    def test_remove_and_reset(self):
        self.ask("enemy has tychus, muradin")
        self.assertEqual(self.ask("remove tychus from the enemy"), "Enemy team: Muradin.")
        self.ask("we're on cursed hollow")
        self.assertEqual(self.ask("reset the draft"), "Draft cleared.")
        self.assertEqual((self.draft.enemy, self.draft.map_slug), ([], None))

    def test_map_by_full_name_nickname_or_misspelling(self):
        for text, expected in (
            ("we're on cursed hollow", "Map: Cursed Hollow."),
            ("the map is dragon shire", "Map: Dragon Shire."),
            ("playing on hollow", "Map: Cursed Hollow."),
            ("map dragon shear", "Map: Dragon Shire."),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.ask(text), expected)
        self.assertIsNone(self.ask("we're on a train"))
        self.assertEqual(self.draft.map_slug, "dragon-shire")  # the failed one changed nothing

    def test_bans_are_recorded(self):
        self.assertEqual(self.ask("they banned muradin"), "Banned: Muradin.")
        self.assertEqual(self.ask("zeratul got banned"), "Banned: Muradin, Zeratul.")

    def test_a_partly_unknown_list_adds_what_it_knows_and_asks_about_the_rest(self):
        answer = self.ask("enemy has tychus and tyra")
        self.assertEqual(answer, "Enemy team: Tychus. Tyrael or Tyrande?")
        self.assertIsNone(self.ask("enemy has banana"))
        self.assertEqual(self.draft.enemy, ["tychus"])

    def test_ambiguous_map_names_are_not_guessed(self):
        self.assertIsNone(self.ask("we're on temple"))


class TestAdvice(IntentCase):
    def test_pick_advice_uses_the_draft(self):
        self.ask("enemy has muradin")
        self.assertEqual(
            self.ask("what should I pick?"),
            "Pick: Tychus, Zeratul. Tychus: counters Muradin and A tier overall.",
        )
        self.assertEqual(self.ask("suggest"), self.ask("what should we pick"))

    def test_pick_advice_can_name_the_enemy_for_one_question_only(self):
        answer = self.ask("what should I pick against muradin")
        self.assertTrue(answer.startswith("Pick: Tychus"))
        self.assertEqual(self.draft.enemy, [])

    def test_ban_advice_and_empty_cases(self):
        self.ask("enemy has uther")
        self.assertTrue(self.ask("what should we ban").startswith("Ban: Muradin"))
        self.draft.reset()
        self.assertEqual(
            self.ask("suggest bans"),
            "Ban: Muradin, Tychus. Muradin: S tier overall and recommended ban.",
        )

    def test_no_clear_answer_is_said_plainly(self):
        self.ask("they banned muradin")
        self.ask("they banned tychus")
        self.assertEqual(self.ask("what should I pick"), "I see no clear pick yet.")

    def test_ordinary_chat_is_left_alone(self):
        for text in ("", "   ", "how is the weather", "what's the time", "tell me a joke"):
            with self.subTest(text=text):
                self.assertIsNone(self.ask(text))


class TestSeam(unittest.TestCase):
    def test_intents_import_no_langchain_or_io(self):
        modules = {m.split(".")[0] for m in imported_modules(module_source("hots/intents.py"))}
        self.assertTrue(
            modules.isdisjoint({"langchain", "langgraph", "httpx", "sqlite3", "asyncio"})
        )


if __name__ == "__main__":
    unittest.main()
