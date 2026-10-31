import unittest

from voicemate.hotkey import Combo, HoldDetector, HotkeyError, parse_combo


class TestParseCombo(unittest.TestCase):
    def test_modifiers_and_a_key_in_any_spelling(self):
        for text in ("cmd+alt+a", "Command+Option+A", " ALT + cmd + a ", "meta+opt+a"):
            with self.subTest(text=text):
                self.assertEqual(parse_combo(text), Combo(frozenset({"cmd", "alt"}), "a"))

    def test_digits_and_space_are_keys(self):
        self.assertEqual(parse_combo("ctrl+shift+7").key, "7")
        self.assertEqual(parse_combo("ctrl+space"), Combo(frozenset({"ctrl"}), "space"))

    def test_a_bad_combination_says_what_a_good_one_looks_like(self):
        for text in ("a", "cmd", "cmd+alt", "cmd+a+b", "cmd+f5", "cmd+", ""):
            with self.subTest(text=text), self.assertRaises(HotkeyError) as caught:
                parse_combo(text)
            self.assertIn("cmd+alt+a", str(caught.exception))


class TestHoldDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HoldDetector(parse_combo("cmd+alt+a"))
        self.both = {"cmd", "alt"}

    def test_the_hold_starts_with_the_key_and_ends_with_its_release(self):
        self.assertTrue(self.detector.key("a", True, self.both))
        self.assertTrue(self.detector.holding)
        self.assertFalse(self.detector.key("a", False, self.both))
        self.assertFalse(self.detector.holding)

    def test_key_repeat_does_not_start_the_hold_again(self):
        self.detector.key("a", True, self.both)
        self.assertIsNone(self.detector.key("a", True, self.both))

    def test_letting_go_of_a_modifier_ends_the_hold(self):
        self.detector.key("a", True, self.both)
        self.assertIsNone(self.detector.modifiers_changed(self.both))
        self.assertFalse(self.detector.modifiers_changed({"cmd"}))
        self.assertIsNone(self.detector.key("a", False, {"cmd"}))  # already over

    def test_other_keys_and_other_modifiers_do_not_count(self):
        self.assertIsNone(self.detector.key("b", True, self.both))
        self.assertIsNone(self.detector.key("a", True, {"cmd"}))
        self.assertIsNone(self.detector.key("a", True, {"cmd", "alt", "shift"}))
        self.assertFalse(self.detector.holding)

    def test_a_release_without_a_hold_says_nothing(self):
        self.assertIsNone(self.detector.key("a", False, self.both))
        self.assertIsNone(self.detector.modifiers_changed(set()))


if __name__ == "__main__":
    unittest.main()
