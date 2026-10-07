import unittest

from voicemate.tts.normalize import speakable


class TestSpeakable(unittest.TestCase):
    def test_markdown_markup_is_removed(self):
        text = "## Title\n**Bold** and *italic* with `code`.\n- first item\n1. second item"
        self.assertEqual(speakable(text), "Title Bold and italic with code. first item second item")

    def test_links_keep_their_label_and_bare_urls_disappear(self):
        text = "See [the paper](https://arxiv.org/abs/1234) or https://example.com/x?y=1 now."
        self.assertEqual(speakable(text), "See the paper or now.")

    def test_code_blocks_are_not_read_aloud(self):
        self.assertEqual(speakable("Run this:\n```python\nprint(1)\n```\nDone."), "Run this: Done.")

    def test_emoji_is_dropped(self):
        self.assertEqual(speakable("Great job 🎉!"), "Great job!")

    def test_persona_pronunciation_is_applied_as_whole_word(self):
        pron = {"Ava": "Éva"}
        self.assertEqual(speakable("Szia, Ava vagyok.", pron), "Szia, Éva vagyok.")
        self.assertEqual(speakable("Avalanche", pron), "Avalanche")

    def test_arithmetic_asterisks_are_not_emphasis(self):
        self.assertEqual(speakable("It is 2*3*4 = 24."), "It is 2*3*4 = 24.")

    def test_empty_input(self):
        self.assertEqual(speakable("   "), "")


if __name__ == "__main__":
    unittest.main()
