import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

INTEGRATION = bool(os.environ.get("VOICEMATE_INTEGRATION")) and sys.platform == "darwin"


@unittest.skipUnless(INTEGRATION, "needs the Parakeet checkpoint and macOS `say`")
class TestParakeetRecognizer(unittest.TestCase):  # pragma: no cover - integration only
    @classmethod
    def setUpClass(cls):
        from voicemate.asr.parakeet import ParakeetRecognizer

        cls.asr = ParakeetRecognizer()

    def _say(self, voice: str, text: str) -> np.ndarray:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(
                ["say", "-v", voice, "-o", tmp.name, "--data-format=LEI16@16000", text], check=True
            )
            audio, _ = sf.read(tmp.name, dtype="float32")
        return audio

    def test_english(self):
        text = self.asr.transcribe_text(self._say("Samantha", "What will the weather be like?"))
        self.assertIn("weather", text.lower())

    def test_hungarian(self):
        text = self.asr.transcribe_text(self._say("Tünde", "Köszönöm szépen, ennyi volt mára."))
        self.assertIn("köszönöm", text.lower())

    def test_too_short_input_is_empty(self):
        self.assertEqual(self.asr.transcribe_text(np.zeros(100, dtype=np.float32)), "")


if __name__ == "__main__":
    unittest.main()
