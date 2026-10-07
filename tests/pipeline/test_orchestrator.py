import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver

from tests.helpers import (
    ScriptedRecognizer,
    ToneSynthesizer,
    make_scripted_chat_model,
    make_tool_context,
    tool_call,
)
from voicemate.agent.graph import build_graph
from voicemate.agent.tools import build_tools
from voicemate.audio.codec import SAMPLE_RATE, float32_to_pcm16
from voicemate.audio.vad import EnergyVAD
from voicemate.events import (
    AudioEvent,
    ConfirmEvent,
    ErrorEvent,
    InterruptEvent,
    Lane,
    LaneState,
    MetricsEvent,
    ReminderEvent,
    ReplyEvent,
    StatusEvent,
    TokenEvent,
    TranscriptEvent,
)
from voicemate.pipeline.orchestrator import (
    APOLOGIES,
    FILLERS,
    INTERRUPTED_MARK,
    INTERRUPTED_TOOL_RESULT,
    PARTIAL_WINDOW_S,
    ChatLine,
    VoiceSession,
    _Pending,
    turns_log_path,
)
from voicemate.runtime import Runtime
from voicemate.tts.voices import VoiceBank


def utterance_pcm(speech_s: float = 0.8) -> list[bytes]:
    t = np.arange(int(speech_s * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.concatenate(
        [np.zeros(4000), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(int(0.9 * SAMPLE_RATE))]
    ).astype(np.float32)
    return [float32_to_pcm16(audio[i : i + 320]) for i in range(0, len(audio), 320)]


class SessionTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tool_context(self.tmp.name)
        self.ctx.config.vad.min_silence_ms = 300
        # Partial transcripts would consume scripted transcripts; only one test enables them.
        self.ctx.config.asr.partial_interval_s = 1000.0
        self.voice = ToneSynthesizer(22050)

    async def asyncTearDown(self):
        await self.session.close()
        await self.runtime.aclose()
        await self.ctx.http.aclose()
        self.tmp.cleanup()

    def make_session(
        self, responses, transcripts=("Mennyi az idő?",), delay=0.0, repeat=False, token_delay=0.0
    ):
        self.llm = make_scripted_chat_model(responses, delay=delay, token_delay=token_delay)
        graph = build_graph(
            self.llm, build_tools(self.ctx), self.ctx.config, self.ctx.memory, InMemorySaver()
        )
        self.runtime = Runtime(
            config=self.ctx.config,
            recognizer=ScriptedRecognizer(list(transcripts), repeat_last=repeat),
            voices=VoiceBank({"hu": self.voice, "en": self.voice}),
            vad_factory=EnergyVAD,
            graph=graph,
            memory=self.ctx.memory,
            notes=self.ctx.notes,
        )
        self.session = VoiceSession(self.runtime, thread_id="test")
        self.events = self.session.bus.subscribe()
        return self.session

    def drain(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out

    async def speak_half(self):
        for frame in utterance_pcm()[:40]:  # stopped in the middle of a sentence
            await self.session.feed_audio(frame)
        self.assertTrue(self.session.segmenter.in_speech)

    async def speak(self, speech_s=0.8):
        for frame in utterance_pcm(speech_s):
            await self.session.feed_audio(frame)
            await asyncio.sleep(0)

    async def settle(self):
        await asyncio.wait_for(self.session.wait_idle(), timeout=10)


class TestVoiceTurn(SessionTestCase):
    async def test_spoken_question_is_answered_with_audio(self):
        self.make_session(["Most este fél tíz van."])
        await self.speak()
        await self.settle()
        events = self.drain()
        final = [e for e in events if isinstance(e, TranscriptEvent) and e.final]
        self.assertEqual(final[0].text, "Mennyi az idő?")
        self.assertEqual(final[0].lang, "hu")
        self.assertIn("fél tíz", "".join(e.text for e in events if isinstance(e, TokenEvent)))
        self.assertTrue(any(isinstance(e, AudioEvent) and e.sample_rate == 22050 for e in events))
        reply = [e for e in events if isinstance(e, ReplyEvent)][-1]
        self.assertFalse(reply.interrupted)
        metrics = [e for e in events if isinstance(e, MetricsEvent)][0].metrics
        self.assertIn("first_audio_s", metrics)
        self.assertIn("asr_s", metrics)
        logged = turns_log_path(self.ctx.config.data_path).read_text().splitlines()
        self.assertEqual(json.loads(logged[-1])["lang"], "hu")
        lanes = [(e.lane, e.state) for e in events if isinstance(e, StatusEvent)]
        for expected in [
            (Lane.LISTENING, LaneState.ACTIVE),
            (Lane.LISTENING, LaneState.DONE),
            (Lane.TRANSCRIBING, LaneState.DONE),
            (Lane.RECALLING, LaneState.DONE),
            (Lane.THINKING, LaneState.DONE),
            (Lane.GENERATING, LaneState.ACTIVE),
            (Lane.SYNTHESIZING, LaneState.DONE),
        ]:
            self.assertIn(expected, lanes)
        await asyncio.sleep(0.05)  # episode is stored in the background
        self.assertEqual(self.ctx.memory.stats()["episodes"], 1)

    async def test_noise_without_words_is_ignored(self):
        self.make_session(["OK"], transcripts=["..."])
        await self.speak()
        await self.settle()
        self.assertFalse(self.session.busy)

    async def test_partial_transcripts_during_long_speech(self):
        self.make_session(["Ok."], transcripts=["long question"], repeat=True)
        self.ctx.config.asr.partial_interval_s = 0.5
        await self.speak(speech_s=2.0)
        await self.settle()
        partials = [e for e in self.drain() if isinstance(e, TranscriptEvent) and not e.final]
        self.assertGreaterEqual(len(partials), 1)

    async def test_microphone_off_mid_sentence_answers_what_was_heard(self):
        self.make_session(["Here I am."], transcripts=["Are you there?"])
        await self.speak_half()
        self.session.microphone_stopped()
        self.assertFalse(self.session.segmenter.in_speech)
        await self.settle()
        finals = [e.text for e in self.drain() if isinstance(e, TranscriptEvent) and e.final]
        self.assertEqual(finals, ["Are you there?"])
        self.assertEqual(self.voice.calls, ["Here I am."])

    async def test_microphone_off_in_silence_just_stops_listening(self):
        self.make_session(["x"])
        self.session.microphone_stopped()
        await self.settle()
        stopped = [e for e in self.drain() if isinstance(e, StatusEvent)]
        self.assertEqual((stopped[-1].lane, stopped[-1].state), (Lane.LISTENING, LaneState.IDLE))
        self.assertEqual(self.voice.calls, [])


class TestTextTurns(SessionTestCase):
    async def test_typed_english_question(self):
        self.make_session(["Hello Adam."])
        await self.session.submit_text("What is the time?")
        await self.settle()
        events = self.drain()
        self.assertEqual([e.lang for e in events if isinstance(e, TranscriptEvent)], ["en"])
        self.assertEqual(self.voice.calls, ["Hello Adam."])

    async def test_network_tool_triggers_spoken_filler(self):
        self.make_session([tool_call("web_search", {"query": "moe"}), "Találtam valamit."])
        await self.session.submit_text("Keress rá a mixture of experts témára")
        await self.settle()
        self.assertEqual(self.voice.calls[0], FILLERS["hu"])
        lanes = [(e.lane, e.detail) for e in self.drain() if isinstance(e, StatusEvent)]
        self.assertIn((Lane.TOOL, "web_search"), lanes)

    async def test_failure_is_reported_and_apologized(self):
        self.make_session([RuntimeError("ollama unreachable")])
        await self.session.submit_text("Szia, mi a helyzet?")
        await self.settle()
        errors = [e for e in self.drain() if isinstance(e, ErrorEvent)]
        self.assertIn("ollama unreachable", errors[0].message)
        self.assertEqual(self.voice.calls, [APOLOGIES["hu"]])

    async def test_timeout(self):
        self.make_session(["too late"], delay=1.0)
        self.ctx.config.llm.turn_timeout_s = 0.05
        await self.session.submit_text("Hello, what is up?")
        await self.settle()
        errors = [e for e in self.drain() if isinstance(e, ErrorEvent)]
        self.assertIn("timed out", errors[0].message)


class TestBargeIn(SessionTestCase):
    async def test_speech_during_playback_interrupts(self):
        self.make_session(["x"])
        self.session.playback_started()
        self.assertTrue(self.session.segmenter.strict)
        await self.speak()
        events = self.drain()
        self.assertTrue(any(isinstance(e, InterruptEvent) for e in events))
        self.assertFalse(self.session._speaking)

    async def test_cut_off_question_is_merged_with_continuation(self):
        self.make_session(["Answer."], transcripts=["first part", "second part"], delay=0.5)
        await self.speak()
        # wait until the first turn is running (blocked in the slow model), then continue talking
        for _ in range(100):
            if self.session.busy:
                break
            await asyncio.sleep(0.01)
        await self.speak()
        await self.settle()
        finals = [
            (e.text, e.merged) for e in self.drain() if isinstance(e, TranscriptEvent) and e.final
        ]
        self.assertEqual(finals, [("first part", False), ("first part second part", True)])
        state = await self.runtime.graph.aget_state({"configurable": {"thread_id": "test"}})
        humans = [m.content for m in state.values["messages"] if m.type == "human"]
        self.assertEqual(humans, ["first part second part"])

    async def test_playback_end_resets_strict_mode(self):
        self.make_session(["x"])
        self.session.playback_started()
        self.session.playback_ended()
        self.assertFalse(self.session.segmenter.strict)
        self.assertFalse(self.session.busy)


class TestInterruptedHistory(SessionTestCase):
    async def history(self):
        state = await self.runtime.graph.aget_state({"configurable": {"thread_id": "test"}})
        return state.values.get("messages", [])

    async def wait_for_tokens(self):
        for _ in range(200):
            if any(isinstance(e, TokenEvent) for e in list(self.events._queue)):
                return
            await asyncio.sleep(0.01)
        self.fail("no tokens streamed")

    async def test_partial_reply_is_kept_and_marked(self):
        long_reply = " ".join(f"word{i}" for i in range(60)) + "."
        self.make_session([long_reply, "Sure, the short version."], token_delay=0.02)
        await self.session.submit_text("Tell me a long story please")
        await self.wait_for_tokens()
        await self.session.submit_text("Actually make it short")
        await self.settle()
        messages = await self.history()
        kinds = [m.type for m in messages]
        self.assertEqual(kinds, ["human", "ai", "human", "ai"])
        self.assertTrue(messages[1].content.endswith(INTERRUPTED_MARK))
        self.assertIn("word0", messages[1].content)
        self.assertNotIn("word59", messages[1].content)
        # The model saw the interrupted reply when answering the follow-up.
        seen = "\n".join(str(m.content) for m in self.llm.seen[-1])
        self.assertIn(INTERRUPTED_MARK, seen)

    async def test_cancelled_tool_call_gets_a_result(self):
        self.make_session(
            [tool_call("get_weather", {"location": "Gyöngyös"}), "Never reached.", "Hello!"],
            delay=0.3,
        )
        await self.session.submit_text("What is the weather in Gyöngyös?")
        for _ in range(200):  # wait until the tool call is in the checkpointed history
            if any(getattr(m, "tool_calls", None) for m in await self.history()):
                break
            await asyncio.sleep(0.01)
        await self.session.submit_text("Never mind, just say hello")
        await self.settle()
        messages = await self.history()
        calls = [c["id"] for m in messages if m.type == "ai" for c in m.tool_calls]
        results = [m.tool_call_id for m in messages if m.type == "tool"]
        self.assertEqual(sorted(calls), sorted(results))
        self.assertIn(INTERRUPTED_TOOL_RESULT, [m.content for m in messages if m.type == "tool"])


class TestRestore(SessionTestCase):
    async def test_history_shows_what_was_said(self):
        long_reply = " ".join(f"word{i}" for i in range(60)) + "."
        self.make_session(
            [tool_call("get_time", {}), "It is noon.", long_reply, "Ok."], token_delay=0.01
        )
        await self.session.submit_text("What time is it?")
        await self.settle()
        self.drain()
        await self.session.submit_text("Tell me a story")
        for _ in range(200):
            if any(isinstance(e, TokenEvent) for e in list(self.events._queue)):
                break
            await asyncio.sleep(0.01)
        await self.session.interrupt()
        await self.settle()
        fresh = VoiceSession(self.runtime, thread_id="test")
        lines = await fresh.restore()
        self.assertEqual(
            [(line.role, line.text) for line in lines[:2]],
            [("user", "What time is it?"), ("assistant", "It is noon.")],
        )
        self.assertEqual(lines[2], ChatLine("user", "Tell me a story"))
        self.assertTrue(lines[3].interrupted)
        self.assertNotIn(INTERRUPTED_MARK, lines[3].text)
        self.assertIsNone(fresh.confirmation)

    async def test_new_conversation_is_empty(self):
        self.make_session(["x"])
        self.assertEqual(await self.session.restore(), [])


class TestHold(SessionTestCase):
    """Hold until I'm finished: pauses do not end the turn until the user releases."""

    def finals(self, events):
        return [(e.text, e.merged) for e in events if isinstance(e, TranscriptEvent) and e.final]

    async def test_pauses_are_collected_into_one_question(self):
        self.make_session(["One answer."], transcripts=["First thought.", "Second thought."])
        await self.session.hold()
        self.assertTrue(self.session.holding)
        await self.speak()
        await self.speak()
        await self.session.submit_text("And a typed one.")
        await self.settle()
        self.assertEqual(self.llm.seen, [])  # nothing was sent to the model yet
        await self.session.release()
        await self.settle()
        self.assertFalse(self.session.holding)
        question = "First thought. Second thought. And a typed one."
        self.assertEqual(
            self.finals(self.drain()),
            [
                ("First thought.", False),
                ("First thought. Second thought.", True),
                (question, True),
                (question, True),  # the turn itself replaces the same bubble
            ],
        )
        self.assertEqual(len(self.llm.seen), 1)
        self.assertTrue(str(self.llm.seen[0][-1].content).endswith(question))
        self.assertEqual(self.voice.calls, ["One answer."])

    async def test_release_includes_the_sentence_still_being_spoken(self):
        self.make_session(["Ok."], transcripts=["Half a sentence"])
        await self.session.hold()
        await self.speak_half()
        await self.session.release()
        await self.settle()
        self.assertTrue(str(self.llm.seen[0][-1].content).endswith("Half a sentence"))

    async def test_microphone_off_releases_the_hold(self):
        self.make_session(["Ok."], transcripts=["Done talking"])
        await self.session.hold()
        await self.speak_half()
        self.session.microphone_stopped()
        await self.settle()
        self.assertFalse(self.session.holding)
        self.assertEqual(self.voice.calls, ["Ok."])

    async def test_hold_stops_the_current_answer(self):
        long_reply = " ".join(f"word{i}" for i in range(60)) + "."
        self.make_session([long_reply], token_delay=0.02)
        await self.session.submit_text("Tell me a story")
        for _ in range(200):
            if any(isinstance(e, TokenEvent) for e in list(self.events._queue)):
                break
            await asyncio.sleep(0.01)
        await self.session.hold()
        await self.settle()
        replies = [e for e in self.drain() if isinstance(e, ReplyEvent)]
        self.assertTrue(replies[-1].interrupted)
        self.assertFalse(self.session.busy)

    async def test_release_without_words_does_nothing(self):
        self.make_session(["x"])
        await self.session.hold()
        await self.session.release()
        await self.session.release()  # releasing twice is harmless
        await self.settle()
        self.assertEqual(self.llm.seen, [])


class FailingRecognizer:
    def transcribe_text(self, audio):
        raise RuntimeError("ASR crashed")


class TestTurnDiscipline(SessionTestCase):
    async def test_new_input_cancels_the_running_turn(self):
        self.make_session(["The answer."], delay=0.3)
        await self.session.submit_text("first question")
        await asyncio.sleep(0.05)
        await self.session.submit_text("second question")
        await self.settle()
        replies = [e.text.strip() for e in self.drain() if isinstance(e, ReplyEvent) and e.text]
        self.assertEqual(replies, ["The answer."])
        # Only one model call completed, and it was answering the second question.
        last_user = [m for m in self.llm.seen[-1] if m.type == "human"][-1]
        self.assertTrue(str(last_user.content).endswith("second question"))

    async def test_stale_cut_off_question_is_not_merged(self):
        self.make_session(["Ok."], transcripts=["new question"])
        self.session._pending = _Pending("gone", "old question", time.monotonic() - 60)
        await self.speak()
        await self.settle()
        finals = [e.text for e in self.drain() if isinstance(e, TranscriptEvent) and e.final]
        self.assertEqual(finals, ["new question"])

    async def test_asr_failure_is_reported(self):
        self.make_session(["x"])
        self.runtime.recognizer = FailingRecognizer()
        await self.speak()
        await self.settle()
        errors = [e for e in self.drain() if isinstance(e, ErrorEvent)]
        self.assertIn("ASR crashed", errors[0].message)
        self.assertFalse(self.session.busy)

    async def test_long_speech_partials_cover_only_recent_audio(self):
        self.make_session(["Ok."], transcripts=["words"], repeat=True)
        self.ctx.config.asr.partial_interval_s = 0.5
        await self.speak(speech_s=PARTIAL_WINDOW_S + 2)
        await self.settle()
        partials = [e.text for e in self.drain() if isinstance(e, TranscriptEvent) and not e.final]
        self.assertTrue(any(p.startswith("…") for p in partials))


class TestConfirmation(SessionTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.target = Path(self.tmp.name) / "Documents" / "plan.md"
        self.target.write_text("old")

    def ask_to_overwrite(self, reply):
        call = tool_call("write_file", {"path": str(self.target), "content": "new"})
        self.make_session([call, reply])

    async def test_question_is_shown_spoken_and_answered_by_button(self):
        self.ask_to_overwrite("Replaced.")
        await self.session.submit_text("Save the plan")
        await self.settle()
        confirms = [e for e in self.drain() if isinstance(e, ConfirmEvent)]
        self.assertTrue(confirms[-1].open)
        self.assertIn("already exists", self.voice.calls[-1])
        self.assertIsNotNone(self.session.confirmation)
        await self.session.answer_confirmation("yes")
        await self.settle()
        self.assertEqual(self.target.read_text(), "new")
        self.assertIsNone(self.session.confirmation)
        self.assertFalse([e for e in self.drain() if isinstance(e, ConfirmEvent)][-1].open)

    async def test_spoken_no_keeps_the_file(self):
        self.ask_to_overwrite("Okay, I left it.")
        await self.session.submit_text("Save the plan")
        await self.settle()
        await self.session.submit_text("no, keep the old one")
        await self.settle()
        self.assertEqual(self.target.read_text(), "old")
        self.assertIn("Okay, I left it.", self.voice.calls[-1])

    async def test_reopened_page_restores_the_open_question(self):
        self.ask_to_overwrite("Replaced.")
        await self.session.submit_text("Save the plan")
        await self.settle()
        await self.session.close()
        self.session = VoiceSession(self.runtime, thread_id="test")  # the page was reloaded
        self.events = self.session.bus.subscribe()
        await self.session.restore()
        confirms = [e for e in self.drain() if isinstance(e, ConfirmEvent)]
        self.assertIn("already exists", confirms[-1].question)
        await self.session.answer_confirmation("yes")
        await self.settle()
        self.assertEqual(self.target.read_text(), "new")

    async def test_button_without_question_does_nothing(self):
        self.make_session(["x"])
        await self.session.answer_confirmation("yes")
        self.assertFalse(self.session.busy)


class TestReminders(SessionTestCase):
    async def test_due_reminder_is_announced_once(self):
        self.make_session(["x"])
        self.ctx.notes.add("call the doctor", "reminder", due="2020-01-01T09:00")
        self.assertEqual(await self.session.check_reminders(), 1)
        events = self.drain()
        self.assertTrue(any(isinstance(e, ReminderEvent) for e in events))
        self.assertTrue(any(isinstance(e, AudioEvent) for e in events))
        self.assertEqual(await self.session.check_reminders(), 0)

    async def test_watcher_starts_and_stops(self):
        self.make_session(["x"])
        self.session.start()
        await asyncio.sleep(0.01)
        await self.session.close()
        self.assertTrue(self.session._reminders.done())


if __name__ == "__main__":
    unittest.main()
