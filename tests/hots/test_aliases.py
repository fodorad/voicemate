import tempfile
import unittest
from pathlib import Path

from voicemate.hots.aliases import AliasStore


class TestAliasStore(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "data" / "aliases.toml"
        self.store = AliasStore(self.path)

    def test_a_missing_file_has_no_aliases(self):
        self.assertEqual(self.store.load(), {})

    def test_added_spellings_are_normalized_and_survive_a_reload(self):
        self.store.add("Ksza Latat", "xalatath")
        self.store.add("tichus", "tychus")
        self.assertEqual(
            AliasStore(self.path).load(), {"kszalatat": "xalatath", "tichus": "tychus"}
        )

    def test_a_damaged_file_is_ignored(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("not [toml")
        self.assertEqual(self.store.load(), {})
        self.store.add("tichus", "tychus")
        self.assertEqual(self.store.load(), {"tichus": "tychus"})


if __name__ == "__main__":
    unittest.main()
