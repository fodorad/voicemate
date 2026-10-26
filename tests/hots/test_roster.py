import unittest

from voicemate.hots.roster import Hero, Roster, normalize

HEROES = [
    Hero("zuljin", "Zul'jin", "Ranged Assassin"),
    Hero("li-ming", "Li-Ming"),
    Hero("li-li", "Li Li"),
    Hero("e-t-c", "E.T.C."),
    Hero("the-lost-vikings", "The Lost Vikings"),
    Hero("muradin", "Muradin"),
    Hero("tychus", "Tychus"),
    Hero("tyrael", "Tyrael"),
    Hero("tyrande", "Tyrande"),
]


class TestNormalize(unittest.TestCase):
    def test_case_accents_and_punctuation_vanish(self):
        self.assertEqual(normalize("Zul'jin"), "zuljin")
        self.assertEqual(normalize(" Lúcio! "), "lucio")
        self.assertEqual(normalize("E.T.C."), "etc")


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.roster = Roster(HEROES)

    def resolve(self, text):
        return self.roster.resolve(text)

    def test_names_slugs_and_spoken_aliases_resolve_exactly(self):
        for spoken, slug in [
            ("Zul'jin", "zuljin"),
            ("zul jin", "zuljin"),
            ("zuljin", "zuljin"),
            ("lee ming", "li-ming"),
            ("etc", "e-t-c"),
            ("lost vikings", "the-lost-vikings"),
        ]:
            with self.subTest(spoken=spoken):
                self.assertEqual(self.resolve(spoken).hero.slug, slug)

    def test_a_clear_misspelling_resolves_to_the_closest_hero(self):
        self.assertEqual(self.resolve("Muradan").hero.slug, "muradin")
        self.assertEqual(self.resolve("tychis").hero.slug, "tychus")

    def test_the_start_of_exactly_one_name_resolves(self):
        self.assertEqual(self.resolve("mura").hero.slug, "muradin")
        self.assertIsNone(self.resolve("mu").hero)

    def test_an_ambiguous_name_returns_candidates_not_a_guess(self):
        result = self.resolve("tyra")
        self.assertIsNone(result.hero)
        self.assertEqual({h.slug for h in result.candidates}, {"tyrael", "tyrande"})

    def test_an_unknown_name_returns_nothing(self):
        result = self.resolve("banana")
        self.assertIsNone(result.hero)
        self.assertEqual(result.candidates, ())
        self.assertEqual(self.resolve("  ").candidates, ())

    def test_aliases_of_missing_heroes_are_ignored(self):
        self.assertEqual(len(Roster(HEROES[:1])), 1)
        self.assertIsNone(Roster(HEROES[:1]).resolve("etc").hero)

    def test_lookup_and_iteration(self):
        self.assertEqual(self.roster.get("muradin").name, "Muradin")
        self.assertIsNone(self.roster.get("nobody"))
        self.assertEqual(len(list(self.roster)), len(HEROES))


if __name__ == "__main__":
    unittest.main()
