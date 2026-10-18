import json
import re
import unittest
from pathlib import Path

from voicemate.audio.protocol import (
    CONTROL_MESSAGES,
    MicOff,
    PlaybackEnded,
    PlaybackStarted,
    Stop,
    dump,
    parse,
)

AUDIO_JS = Path(__file__).resolve().parents[2] / "voicemate" / "ui" / "static" / "audio.js"


class TestParse(unittest.TestCase):
    def test_every_message_round_trips(self):
        for message in (MicOff(), PlaybackStarted(), PlaybackEnded(), Stop()):
            with self.subTest(message=message):
                self.assertEqual(parse(dump(message)), message)

    def test_wire_format_is_the_json_the_browser_uses(self):
        self.assertEqual(json.loads(dump(MicOff())), {"type": "mic", "state": "off"})
        self.assertEqual(
            json.loads(dump(PlaybackStarted())), {"type": "playback", "state": "started"}
        )
        self.assertEqual(json.loads(dump(PlaybackEnded())), {"type": "playback", "state": "ended"})
        self.assertEqual(json.loads(dump(Stop())), {"type": "stop"})

    def test_bad_input_is_ignored_not_raised(self):
        for text in (
            "not json",
            '["a list"]',
            "42",
            "{}",
            '{"type": "launch-missiles"}',
            '{"type": "playback"}',
            '{"type": "playback", "state": "sideways"}',
            '{"type": "mic", "state": "on"}',
        ):
            with self.subTest(text=text), self.assertLogs("voicemate.audio.protocol", "WARNING"):
                self.assertIsNone(parse(text))


class TestContractWithBrowser(unittest.TestCase):
    """audio.js and the Python definition must agree on every message."""

    def js_messages(self) -> set[tuple[str, str | None]]:
        source = AUDIO_JS.read_text()
        pairs = re.findall(r'type:\s*"(\w+)"(?:,\s*state:\s*"(\w+)")?', source)
        handled = re.findall(r'message\.type\s*===\s*"(\w+)"', source)
        return {(t, s or None) for t, s in pairs} | {(t, None) for t in handled}

    def test_browser_only_uses_known_messages(self):
        known = {(m.type, m.state) for m in CONTROL_MESSAGES}
        self.assertTrue(self.js_messages())
        self.assertLessEqual(self.js_messages(), known)

    def test_every_known_message_appears_in_the_browser(self):
        known = {(m.type, m.state) for m in CONTROL_MESSAGES}
        self.assertEqual(known - self.js_messages(), set())


if __name__ == "__main__":
    unittest.main()
