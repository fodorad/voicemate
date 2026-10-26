import unittest

from tests.helpers import called_names, imported_modules, module_source, uses_async_syntax
from voicemate.config import ActivationConfig, AssistantConfig, Config
from voicemate.pipeline.activation import Activation

WAKE = Activation("wake_word", ("ava", "eva"))


class TestAlways(unittest.TestCase):
    def test_everything_is_a_command_unchanged(self):
        always = Activation("always", ("ava",))
        self.assertEqual(always.command("what time is it"), "what time is it")
        self.assertFalse(always.needs_wake_word)


class TestWakeWord(unittest.TestCase):
    def test_speech_that_starts_with_the_name_is_a_command_without_it(self):
        for text, expected in [
            ("Ava, what should I pick?", "what should I pick?"),
            ("ava what should I pick", "what should I pick"),
            ("Hey Ava, counter Zul'jin", "counter Zul'jin"),
            ("okay eva: ban suggestions", "ban suggestions"),
            ("  AVA!  reset the draft ", "reset the draft"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(WAKE.command(text), expected)

    def test_everything_else_is_dropped(self):
        for text in [
            "we should meet at nine",
            "I told Ava about it",  # the name must come first
            "avatar looks great",  # a longer word is not the name
            "Ava",  # nothing follows
            "Hey Ava...",
            "",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(WAKE.command(text))

    def test_mode_and_names_come_from_the_config(self):
        config = Config(
            assistant=AssistantConfig(name="Nova"),
            activation=ActivationConfig(mode="wake_word", variants=["nova"]),
        )
        rule = Activation.from_config(config)
        self.assertTrue(rule.needs_wake_word)
        self.assertEqual(rule.command("Nova, hello"), "hello")
        self.assertEqual(rule.command("nova hello"), "hello")
        self.assertIsNone(rule.command("Ava, hello"))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            Activation("sometimes", ("ava",))


class TestPurity(unittest.TestCase):
    def test_the_policy_has_no_io_and_no_event_loop(self):
        source = module_source("pipeline/activation.py")
        self.assertFalse(uses_async_syntax(source))
        self.assertEqual(
            {m.split(".")[0] for m in imported_modules(source)} & {"asyncio", "httpx", "sqlite3"},
            set(),
        )
        self.assertFalse(called_names(source) & {"open", "print"})


if __name__ == "__main__":
    unittest.main()
