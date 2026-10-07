import unittest

from voicemate.agent.tools.arxiv_tools import (
    Paper,
    format_papers,
    normalize_id,
    paper_filename,
    slugify,
)
from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.search import format_results, make_backend
from voicemate.agent.tools.weather import describe, format_forecast
from voicemate.config import SearchConfig


class TestWeatherFormatting(unittest.TestCase):
    def test_forecast_text(self):
        data = {
            "current": {"temperature_2m": 14.2, "weather_code": 3, "wind_speed_10m": 9.0},
            "daily": {
                "time": ["2026-09-28", "2026-09-29"],
                "weather_code": [3, 61],
                "temperature_2m_min": [8.0, 9.5],
                "temperature_2m_max": [17.0, 15.1],
                "precipitation_probability_max": [10, 80],
            },
        }
        text = format_forecast(data, "Gyöngyös")
        self.assertIn("Now: 14.2°C, overcast", text)
        self.assertIn("2026-09-29: light rain, 9.5–15.1°C, precipitation chance 80%", text)

    def test_unknown_codes(self):
        self.assertEqual(describe(None), "unknown")
        self.assertEqual(describe(42), "weather code 42")


class TestArxivHelpers(unittest.TestCase):
    def test_normalize_id(self):
        self.assertEqual(normalize_id("https://arxiv.org/abs/2401.01234v2"), "2401.01234v2")
        self.assertEqual(normalize_id("arXiv:1706.03762"), "1706.03762")
        with self.assertRaises(ToolError):
            normalize_id("not a paper")

    def test_filename(self):
        paper = Paper("1706.03762v7", "Attention Is All You Need!", ["A"], "2017-06-12", "s", "u")
        self.assertEqual(paper_filename(paper), "1706.03762v7-attention-is-all-you-need.pdf")
        self.assertEqual(slugify("???"), "paper")
        self.assertEqual(paper.abs_url, "https://arxiv.org/abs/1706.03762v7")

    def test_format_papers(self):
        row = {
            "arxiv_id": "1",
            "title": "T",
            "authors": list("ABCD"),
            "published": "2020",
            "summary": "S",
        }
        self.assertIn("A, B, C et al.", format_papers([row], "h"))
        self.assertIn("No papers", format_papers([], "h"))


class TestSearchHelpers(unittest.TestCase):
    def test_format_empty(self):
        self.assertIn("No results", format_results([], "h"))

    def test_backend_selection(self):
        self.assertEqual(make_backend(SearchConfig(backend="ddgs"), None).name, "ddgs")
        self.assertEqual(make_backend(SearchConfig(backend="searxng"), None).name, "searxng")
        with self.assertRaises(ValueError):
            make_backend(SearchConfig(backend="google"), None)


if __name__ == "__main__":
    unittest.main()
