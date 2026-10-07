import json
import tempfile
import unittest
from pathlib import Path

from voicemate.report import summarize


class TestReport(unittest.TestCase):
    def write(self, turns):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "turns.jsonl"
        path.write_text("\n".join(json.dumps(t) for t in turns) + "\n\n")
        return path

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_missing_log_is_empty(self):
        report = summarize(Path("/nonexistent/turns.jsonl"))
        self.assertEqual(report.turns, 0)
        self.assertIn("n/a", report.render())

    def test_percentiles_tools_and_breaches(self):
        turns = [{"first_audio_s": 1.0 + i * 0.1, "tools": [], "failed": False} for i in range(10)]
        turns += [{"first_audio_s": 9.0, "tools": ["web_search"], "failed": False}]
        turns += [{"tools": [], "failed": True}]
        report = summarize(self.write(turns))
        self.assertEqual(report.turns, 12)
        self.assertAlmostEqual(report.chat_first_audio[0], 1.45)
        self.assertEqual(report.tool_counts, {"web_search": 1})
        self.assertAlmostEqual(report.error_rate, 1 / 12)
        self.assertEqual(len(report.breaches), 2)  # tool latency and error rate
        text = report.render()
        self.assertIn("web_search 1", text)
        self.assertIn("error rate", text)

    def test_within_budget(self):
        report = summarize(self.write([{"first_audio_s": 1.2, "tools": []}]))
        self.assertEqual(report.breaches, [])
        self.assertIn("none", report.render())


if __name__ == "__main__":
    unittest.main()
