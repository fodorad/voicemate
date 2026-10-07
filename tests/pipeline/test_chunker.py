import unittest

from voicemate.pipeline.chunker import SpeechChunker


def stream(chunker: SpeechChunker, text: str, step: int = 3) -> list[str]:
    """Feed ``text`` in small token-like pieces, then flush."""
    out: list[str] = []
    for i in range(0, len(text), step):
        out.extend(chunker.feed(text[i : i + step]))
    out.extend(chunker.flush())
    return out


class TestSpeechChunker(unittest.TestCase):
    def test_splits_sentences(self):
        chunks = stream(SpeechChunker(min_first_chars=1000), "Hello there. How are you? I am fine!")
        self.assertEqual(chunks, ["Hello there.", "How are you?", "I am fine!"])

    def test_emits_sentence_as_soon_as_next_one_starts(self):
        chunker = SpeechChunker(min_first_chars=1000)
        self.assertEqual(chunker.feed("First sentence. "), [])
        self.assertEqual(chunker.feed("Second"), ["First sentence."])

    def test_decimal_numbers_do_not_split(self):
        chunks = stream(SpeechChunker(min_first_chars=1000), "The value is 3.14 today. Nice.")
        self.assertEqual(chunks, ["The value is 3.14 today.", "Nice."])

    def test_abbreviations_do_not_split(self):
        text = "Pl. a kutya, stb. is jó. Dr. Kovács is here, e.g. with Mr. Smith. Done."
        chunks = stream(SpeechChunker(min_first_chars=1000), text)
        self.assertEqual(
            chunks,
            ["Pl. a kutya, stb. is jó.", "Dr. Kovács is here, e.g. with Mr. Smith.", "Done."],
        )

    def test_hungarian_ordinal_followed_by_lowercase_does_not_split(self):
        chunks = stream(SpeechChunker(min_first_chars=1000), "A 2. helyen végzett. Gratulálok.")
        self.assertEqual(chunks, ["A 2. helyen végzett.", "Gratulálok."])

    def test_first_chunk_splits_early_at_clause(self):
        text = "Well, let me think about that for a moment, because it matters. Then more."
        chunks = stream(SpeechChunker(min_first_chars=20), text)
        self.assertEqual(
            chunks,
            ["Well, let me think about that for a moment,", "because it matters.", "Then more."],
        )

    def test_long_run_without_punctuation_is_capped_at_a_space(self):
        chunks = stream(SpeechChunker(min_first_chars=1000, max_chars=30), "word " * 20)
        self.assertTrue(all(len(c) <= 30 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), ["word"] * 20)

    def test_newlines_are_boundaries(self):
        chunks = stream(SpeechChunker(min_first_chars=1000), "First item\nSecond item\n")
        self.assertEqual(chunks, ["First item", "Second item"])

    def test_flush_returns_nothing_for_whitespace(self):
        chunker = SpeechChunker()
        chunker.feed("   ")
        self.assertEqual(chunker.flush(), [])

    def test_reset_starts_a_new_turn(self):
        chunker = SpeechChunker(min_first_chars=10)
        stream(chunker, "One two three four, five six.")
        chunker.reset()
        self.assertEqual(
            stream(chunker, "Alpha beta gamma, delta."), ["Alpha beta gamma,", "delta."]
        )


if __name__ == "__main__":
    unittest.main()
