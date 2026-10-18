import unittest

from voicemate.agent.tools import EGRESS_TOOLS
from voicemate.asr.base import Transcript
from voicemate.pipeline.reply import APOLOGIES, FILLERS, Reply


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def drain(reply: Reply) -> list[str | None]:
    out = []
    while not reply.queue.empty():
        out.append(reply.queue.get_nowait())
    return out


class TestText(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.reply = Reply("en", ended_at=100.0, clock=self.clock)

    def test_text_accumulates_and_sentences_are_queued_as_they_complete(self):
        self.assertTrue(self.reply.add_text("Hello there Adam, nice to meet you. And "))
        self.assertFalse(self.reply.add_text("more to say."))
        self.assertEqual(self.reply.text, "Hello there Adam, nice to meet you. And more to say.")
        self.assertEqual(drain(self.reply), ["Hello there Adam, nice to meet you."])
        self.assertTrue(self.reply.spoke)

    def test_nothing_is_spoken_before_a_sentence_is_complete(self):
        self.reply.add_text("Hello there")
        self.assertFalse(self.reply.spoke)
        self.assertEqual(drain(self.reply), [])

    def test_finish_flushes_the_rest_and_closes_the_queue(self):
        self.reply.add_text("Hello there")
        self.reply.finish()
        self.assertEqual(drain(self.reply), ["Hello there", None])

    def test_the_first_token_is_recorded_once(self):
        self.assertTrue(self.reply.mark_first_token())
        self.clock.now = 105.0
        self.assertFalse(self.reply.mark_first_token())
        self.assertEqual(self.reply.first_token, 100.0)

    def test_the_first_audio_is_recorded_once(self):
        self.reply.audio_started()
        self.clock.now = 109.0
        self.reply.audio_started()
        self.assertEqual(self.reply.first_audio, 100.0)


class TestTools(unittest.TestCase):
    def setUp(self):
        self.reply = Reply("hu", ended_at=100.0, clock=Clock())

    def test_a_network_tool_before_any_speech_queues_the_filler_once(self):
        egress = sorted(EGRESS_TOOLS)[0]
        self.assertTrue(self.reply.on_tool_start(egress))
        self.reply.on_tool_start(egress)
        self.assertEqual(drain(self.reply), [FILLERS["hu"]])
        self.assertEqual(self.reply.tools, [egress, egress])

    def test_a_local_tool_needs_no_filler(self):
        self.reply.on_tool_start("calculator")
        self.assertEqual(drain(self.reply), [])

    def test_no_filler_once_something_was_said(self):
        self.reply.add_text("Először is, jó kérdés ez, nézzük meg. Aztán")
        drain(self.reply)
        self.reply.on_tool_start(sorted(EGRESS_TOOLS)[0])
        self.assertEqual(drain(self.reply), [])

    def test_an_unknown_language_gets_the_english_filler(self):
        reply = Reply("de", ended_at=0.0, clock=Clock())
        reply.on_tool_start(sorted(EGRESS_TOOLS)[0])
        self.assertEqual(drain(reply), [FILLERS["en"]])

    def test_a_tool_start_counts_as_the_first_token_but_only_once(self):
        self.assertTrue(self.reply.on_tool_start("calculator"))
        self.assertFalse(self.reply.on_tool_start("calculator"))
        self.assertFalse(self.reply.mark_first_token())


class TestFailure(unittest.TestCase):
    def test_a_timeout_apologises_in_the_turn_language(self):
        reply = Reply("hu", ended_at=0.0, clock=Clock())
        self.assertEqual(reply.fail(TimeoutError()), "timed out")
        self.assertTrue(reply.failed)
        self.assertEqual(drain(reply), [APOLOGIES["hu"]])

    def test_other_errors_use_their_message_or_their_type(self):
        self.assertEqual(Reply("en", 0.0, Clock()).fail(ConnectionError("no route")), "no route")
        self.assertEqual(Reply("en", 0.0, Clock()).fail(KeyError()), "KeyError")

    def test_the_apology_falls_back_to_english(self):
        reply = Reply("de", ended_at=0.0, clock=Clock())
        reply.fail(RuntimeError("x"))
        self.assertEqual(drain(reply), [APOLOGIES["en"]])


class TestMetrics(unittest.TestCase):
    def test_timings_are_relative_to_the_end_of_the_users_speech(self):
        clock = Clock(100.0)
        reply = Reply("en", ended_at=100.0, clock=clock)
        clock.now = 101.0
        reply.mark_first_token()
        reply.add_text("Done.")
        clock.now = 102.5
        reply.audio_started()
        reply.on_tool_start("calculator")
        clock.now = 104.0
        transcript = Transcript("hi", "en", 1.234, 0.4567)
        metrics = reply.metrics(transcript)
        self.assertEqual(metrics["lang"], "en")
        self.assertEqual(metrics["tools"], ["calculator"])
        self.assertEqual(metrics["asr_s"], 0.457)
        self.assertEqual(metrics["audio_s"], 1.23)
        self.assertEqual(metrics["first_token_s"], 1.0)
        self.assertEqual(metrics["first_audio_s"], 2.5)
        self.assertEqual(metrics["total_s"], 4.0)
        self.assertEqual(metrics["chars_out"], 5)
        self.assertFalse(metrics["failed"])
        self.assertIn("ts", metrics)

    def test_typed_turns_have_no_asr_numbers_and_silent_turns_no_audio_time(self):
        metrics = Reply("en", ended_at=0.0, clock=Clock(1.0)).metrics(None)
        for key in ("asr_s", "audio_s", "first_token_s", "first_audio_s"):
            self.assertNotIn(key, metrics)
        self.assertEqual(metrics["total_s"], 1.0)

    def test_a_failed_turn_says_so(self):
        reply = Reply("en", ended_at=0.0, clock=Clock())
        reply.fail(RuntimeError("boom"))
        self.assertTrue(reply.metrics(None)["failed"])


if __name__ == "__main__":
    unittest.main()
