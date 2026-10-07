import unittest

from voicemate.agent.hitl import is_affirmative


class TestIsAffirmative(unittest.TestCase):
    def test_yes_answers(self):
        for answer in ("Yes", "yes, go ahead", "Sure.", "okay do it", "Igen, mehet", "persze"):
            with self.subTest(answer=answer):
                self.assertTrue(is_affirmative(answer))

    def test_no_and_unclear_answers(self):
        for answer in ("no", "No, call it summary2", "yes... no wait", "don't", "nem", "", "hmm"):
            with self.subTest(answer=answer):
                self.assertFalse(is_affirmative(answer))


if __name__ == "__main__":
    unittest.main()
