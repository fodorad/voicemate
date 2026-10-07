import os
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ToneSynthesizer
from voicemate.config import Config
from voicemate.models import PiperVoiceId, kokoro_paths, piper_voice_path
from voicemate.tts.voices import VoiceBank


class TestVoiceBank(unittest.TestCase):
    def setUp(self):
        self.hu, self.en = ToneSynthesizer(22050), ToneSynthesizer(24000)
        self.bank = VoiceBank(
            {"hu": self.hu, "en": self.en}, {"hu": {"Ava": "Éva"}}, default_lang="hu"
        )

    def test_language_selects_voice_and_sample_rate(self):
        audio, rate = self.bank.synthesize("Hello there.", "en")
        self.assertEqual(rate, 24000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(self.en.calls, ["Hello there."])

    def test_unknown_language_falls_back_to_default(self):
        _, rate = self.bank.synthesize("Hallo.", "de")
        self.assertEqual(rate, 22050)

    def test_text_is_cleaned_and_pronunciation_applied(self):
        self.bank.synthesize("**Szia**, Ava vagyok: https://x.y", "hu")
        self.assertEqual(self.hu.calls, ["Szia, Éva vagyok:"])

    def test_nothing_speakable_skips_synthesis(self):
        audio, _ = self.bank.synthesize("```code```", "hu")
        self.assertEqual(len(audio), 0)
        self.assertEqual(self.hu.calls, [])

    def test_default_language_must_have_a_voice(self):
        with self.assertRaises(ValueError):
            VoiceBank({"en": self.en}, default_lang="hu")


class TestModelPaths(unittest.TestCase):
    def test_piper_voice_id(self):
        voice = PiperVoiceId.parse("hu_HU-anna-medium")
        self.assertEqual(voice.repo_path, "hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx")
        with self.assertRaises(ValueError):
            PiperVoiceId.parse("anna")

    def test_local_paths_live_under_data_models(self):
        config = Config(data_dir="/tmp/vm")
        self.assertEqual(
            piper_voice_path(config, "hu_HU-anna-medium"),
            Path("/tmp/vm/models/piper/hu_HU-anna-medium.onnx"),
        )
        self.assertEqual(kokoro_paths(config)[0].parent, Path("/tmp/vm/models/kokoro"))


@unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION"), "downloads voices")
class TestRealVoices(unittest.TestCase):  # pragma: no cover - integration only
    def test_piper_and_kokoro_render_audio(self):
        from voicemate import models
        from voicemate.config import load_config
        from voicemate.tts.kokoro import KokoroSynthesizer
        from voicemate.tts.piper import PiperSynthesizer

        config = load_config()
        piper = PiperSynthesizer(models.ensure_piper_voice(config, "hu_HU-anna-medium"))
        audio = piper.synthesize("Szia, miben segíthetek?")
        self.assertGreater(len(audio) / piper.sample_rate, 0.5)
        kokoro = KokoroSynthesizer(*models.ensure_kokoro(config))
        audio = kokoro.synthesize("Hello, how can I help?")
        self.assertGreater(len(audio) / kokoro.sample_rate, 0.5)
        self.assertIn("h", kokoro.phonemize("hello"))

    def test_download_is_idempotent(self):
        from voicemate import models

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(data_dir=tmp)
            first = models.ensure_piper_voice(config, "hu_HU-berta-medium")
            mtime = first.stat().st_mtime
            self.assertEqual(models.ensure_piper_voice(config, "hu_HU-berta-medium"), first)
            self.assertEqual(first.stat().st_mtime, mtime)
            self.assertTrue(first.with_suffix(".onnx.json").exists())


if __name__ == "__main__":
    unittest.main()
