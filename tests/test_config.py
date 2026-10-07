import tempfile
import unittest
from pathlib import Path

from voicemate.config import PROJECT_DIR, Config, TTSVoiceConfig, load_config


class TestLoadConfig(unittest.TestCase):
    def test_repository_default_file_loads(self):
        config = load_config(PROJECT_DIR / "config" / "voicemate.toml")
        self.assertEqual(config.assistant.name, "Ava")
        self.assertEqual(config.assistant.default_language, "en")
        self.assertEqual(
            config.tts["hu"], TTSVoiceConfig(engine="piper", voice="hu_HU-anna-medium")
        )
        self.assertEqual(config.tts["en"].engine, "kokoro")
        self.assertAlmostEqual(config.location.latitude, 47.7826)
        self.assertEqual(config.files.roots, ["~/Documents", "~/Downloads"])

    def test_missing_sections_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text('[llm]\nprofile = "qwen"\n')
            config = load_config(path)
        self.assertEqual(config.llm.profile, "qwen")
        self.assertEqual(config.llm.model, Config().llm.profiles["qwen"])
        self.assertEqual(config.llm.max_tool_rounds, Config().llm.max_tool_rounds)
        self.assertEqual(config.vad, Config().vad)

    def test_missing_default_file_means_built_in_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(env={}, default_path=Path(tmp) / "absent.toml")
        self.assertEqual(config, Config())

    def test_missing_explicit_file_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_config(Path("/nonexistent/voicemate.toml"))

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("[llm]\nmodle = 'typo'\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_environment_variable_selects_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text('[assistant]\nname = "Mate"\n')
            config = load_config(env={"VOICEMATE_CONFIG": str(path)})
        self.assertEqual(config.assistant.name, "Mate")


class TestPaths(unittest.TestCase):
    def test_relative_data_dir_resolves_against_project(self):
        config = Config(data_dir="data")
        self.assertEqual(config.data_path, PROJECT_DIR / "data")

    def test_user_paths_are_expanded(self):
        config = Config()
        config.files.roots = ["~/Documents"]
        self.assertEqual(config.files.root_paths, [Path.home() / "Documents"])
        self.assertTrue(config.files.papers_path.is_absolute())


if __name__ == "__main__":
    unittest.main()
