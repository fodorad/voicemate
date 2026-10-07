import unittest

from voicemate.agent.model_select import choose_model, estimated_load_gb, normalize_name
from voicemate.config import LLMConfig


def llm(**overrides) -> LLMConfig:
    config = LLMConfig(
        profile="quality",
        profiles={"fast": "gemma4:e4b", "quality": "qwen3.8:27b-mlx"},
        fallback_profile="fast",
        reserve_gb=5.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


INSTALLED = {"gemma4:e4b": 6.6, "qwen3.8:27b-mlx": 18.2}


class TestChooseModel(unittest.TestCase):
    def test_preferred_profile_when_it_fits(self):
        choice = choose_model(llm(), INSTALLED, available_gb=40.0)
        self.assertEqual(
            (choice.profile, choice.model, choice.fits), ("quality", "qwen3.8:27b-mlx", True)
        )
        self.assertFalse(choice.fell_back)

    def test_falls_back_when_memory_is_short(self):
        choice = choose_model(llm(), INSTALLED, available_gb=16.0)
        self.assertEqual((choice.profile, choice.model), ("fast", "gemma4:e4b"))
        self.assertTrue(choice.fell_back)
        self.assertTrue(choice.fits)
        self.assertIn("qwen3.8:27b-mlx", choice.reason)

    def test_falls_back_when_model_is_not_installed(self):
        choice = choose_model(llm(), {"gemma4:e4b": 6.6}, available_gb=64.0)
        self.assertEqual(choice.model, "gemma4:e4b")
        self.assertIn("not installed", choice.reason)

    def test_nothing_fits_keeps_smallest_and_warns(self):
        choice = choose_model(llm(), INSTALLED, available_gb=4.0)
        self.assertEqual(choice.model, "gemma4:e4b")
        self.assertFalse(choice.fits)

    def test_nothing_installed_returns_preferred_unverified(self):
        choice = choose_model(llm(), {}, available_gb=64.0)
        self.assertEqual(choice.model, "qwen3.8:27b-mlx")
        self.assertFalse(choice.fits)
        self.assertIn("ollama pull", choice.reason)

    def test_guard_disabled_trusts_the_profile(self):
        choice = choose_model(llm(memory_guard=False), INSTALLED, available_gb=1.0)
        self.assertEqual(choice.model, "qwen3.8:27b-mlx")

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_model(llm(profile="turbo"), INSTALLED, available_gb=64.0)


class TestHelpers(unittest.TestCase):
    def test_names_without_tag_mean_latest(self):
        self.assertEqual(normalize_name("gemma4"), "gemma4:latest")
        self.assertEqual(normalize_name("gemma4:e4b"), "gemma4:e4b")

    def test_load_estimate_grows_with_size(self):
        self.assertGreater(estimated_load_gb(18.2), estimated_load_gb(6.6))
        self.assertGreater(estimated_load_gb(6.6), 6.6)

    def test_llm_model_follows_profile(self):
        config = llm(profile="fast")
        self.assertEqual(config.model, "gemma4:e4b")


if __name__ == "__main__":
    unittest.main()
