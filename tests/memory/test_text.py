import unittest

from voicemate.memory.text import chunk_text, ttl_days_for


class TestChunkText(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("One paragraph."), ["One paragraph."])

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(chunk_text("  \n "), [])

    def test_long_text_respects_max_chars_and_keeps_all_words(self):
        paragraphs = [f"Paragraph {i}. " + "word " * 60 for i in range(6)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_chars=400)
        self.assertTrue(all(len(c) <= 400 for c in chunks))
        joined = " ".join(chunks)
        for i in range(6):
            self.assertIn(f"Paragraph {i}.", joined)

    def test_single_huge_paragraph_is_split_at_sentences(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(100))
        chunks = chunk_text(text, max_chars=300)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_unbreakable_run_is_hard_cut(self):
        chunks = chunk_text("x" * 1000, max_chars=300)
        self.assertEqual("".join(chunks), "x" * 1000)


class TestTTL(unittest.TestCase):
    def test_time_sensitive_queries_expire_after_a_day(self):
        for query in ("latest news on AI", "ML állások Budapest", "legújabb hírek", "jobs today"):
            with self.subTest(query=query):
                self.assertEqual(ttl_days_for(query, "web"), 1)

    def test_general_web_and_papers(self):
        self.assertEqual(ttl_days_for("how does LoRA work", "web"), 30)
        self.assertEqual(ttl_days_for("mixture of experts", "arxiv"), 365)
        self.assertEqual(ttl_days_for("anything", "paper"), 365)


if __name__ == "__main__":
    unittest.main()
