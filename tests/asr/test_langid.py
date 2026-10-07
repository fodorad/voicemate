import unittest

from voicemate.asr.langid import LanguageDetector


class TestLanguageDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = LanguageDetector()

    def test_plain_sentences(self):
        cases = {
            "Mennyi az idő most?": "hu",
            "Milyen idő lesz holnap Gyöngyösön?": "hu",
            "What time is it right now?": "en",
            "Find machine learning engineer jobs in Budapest.": "en",
            "Köszönöm szépen, ennyi volt mára.": "hu",
        }
        for text, lang in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.detector.detect(text), lang)

    def test_hungarian_with_english_technical_terms(self):
        for text in (
            "Keress rá a transformer architecture-ökre",
            "Mi az a LoRA fine tuning?",
            "Foglald össze a letöltött PDF file tartalmát.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.detector.detect(text), "hu")

    def test_english_mentioning_hungarian_place(self):
        self.assertEqual(self.detector.detect("What is the weather in Gyöngyös tomorrow?"), "en")

    def test_short_ambiguous_utterance_keeps_previous_language(self):
        self.assertEqual(self.detector.detect("ok", previous="hu"), "hu")
        self.assertEqual(self.detector.detect("ok", previous="en"), "en")
        self.assertEqual(self.detector.detect("", previous="hu"), "hu")

    def test_statistical_fallback_without_function_words(self):
        self.assertEqual(self.detector.detect("machine learning engineer salary"), "en")

    def test_default_when_nothing_is_known(self):
        self.assertIn(self.detector.detect("ok"), ("hu", "en"))


if __name__ == "__main__":
    unittest.main()
