import contextlib
import io
import unittest

from voicemate.__main__ import build_parser, main, render_event
from voicemate.events import ConfirmEvent, ErrorEvent, TokenEvent, ToolEvent


class TestCli(unittest.TestCase):
    def test_report_command_prints_a_summary(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["report"])
        self.assertIn("Turns:", out.getvalue())
        self.assertIn("Budget breaches:", out.getvalue())

    def test_unknown_profile_is_rejected_early(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["chat", "--profile", "turbo"])

    def test_parser_defaults(self):
        args = build_parser().parse_args(["bench"])
        self.assertEqual(args.what, "all")
        self.assertIsNone(build_parser().parse_args([]).command)


class TestRenderEvent(unittest.TestCase):
    def test_text_chat_rendering(self):
        out = io.StringIO()
        for event in (
            TokenEvent("Hello"),
            ToolEvent("web_search", "end", source="web", summary="moe"),
            ToolEvent("web_search", "start"),
            ConfirmEvent("Replace plan.md?"),
            ErrorEvent("Turn failed: timed out"),
        ):
            render_event(event, out)
        text = out.getvalue()
        self.assertIn("Hello", text)
        self.assertIn("[web_search: web, moe]", text)
        self.assertIn("? Replace plan.md?", text)
        self.assertIn("! Turn failed: timed out", text)
        self.assertEqual(text.count("web_search"), 1)


if __name__ == "__main__":
    unittest.main()
