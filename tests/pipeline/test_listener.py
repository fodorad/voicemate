import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import EnergyVAD
from voicemate.audio.codec import SAMPLE_RATE, float32_to_pcm16
from voicemate.audio.vad import SpeechEnd, SpeechStart
from voicemate.config import VADConfig
from voicemate.events import LaneState
from voicemate.pipeline.listener import PARTIAL_WINDOW_S, Listener, ListenerCallbacks, Utterance


def frames(speech_s: float = 0.8, silence_s: float = 0.9) -> list[bytes]:
    t = np.arange(int(speech_s * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.concatenate(
        [np.zeros(4000), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(int(silence_s * SAMPLE_RATE))]
    ).astype(np.float32)
    return [float32_to_pcm16(audio[i : i + 320]) for i in range(0, len(audio), 320)]


class Recording:
    """Callbacks that record everything, plus a controllable recognizer."""

    def __init__(self, texts=("hello there",)) -> None:
        self.texts = list(texts)
        self.gate: asyncio.Event | None = None
        self.fail = False
        self.seen_audio: list[int] = []
        self.partials: list[str] = []
        self.transcribing: list[LaneState] = []
        self.heard: list[Utterance] = []
        self.errors: list[Exception] = []
        self.slow_utterance = False

    async def transcribe(self, audio: np.ndarray) -> str:
        self.seen_audio.append(len(audio))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("ASR crashed")
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]

    async def on_utterance(self, utterance: Utterance) -> None:
        if self.slow_utterance:
            await asyncio.sleep(0.05)
        self.heard.append(utterance)

    def callbacks(self) -> ListenerCallbacks:
        return ListenerCallbacks(
            on_partial=self.partials.append,
            on_transcribing=self.transcribing.append,
            on_utterance=self.on_utterance,
            on_error=self.errors.append,
        )


class ListenerTestCase(unittest.IsolatedAsyncioTestCase):
    def make(self, rec: Recording, **kwargs) -> Listener:
        kwargs.setdefault("partial_interval_s", 1000.0)
        self.tasks: list[asyncio.Task] = []

        def spawn(coro):
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            return task

        listener = Listener(
            EnergyVAD(),
            VADConfig(min_silence_ms=300),
            rec.transcribe,
            spawn,
            rec.callbacks(),
            **kwargs,
        )
        self.addAsyncCleanup(self.cleanup)
        return listener

    async def cleanup(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def feed_all(self, listener: Listener, pcm: list[bytes]) -> list:
        events = []
        for frame in pcm:
            events.extend(listener.feed(frame))
        return events


class TestUtterances(ListenerTestCase):
    async def test_speech_yields_start_end_and_a_transcribed_utterance(self):
        rec = Recording(["what time is it"])
        listener = self.make(rec)
        events = self.feed_all(listener, frames())
        self.assertEqual([type(e) for e in events], [SpeechStart, SpeechEnd])
        await listener.settled()
        [heard] = rec.heard
        self.assertEqual(heard.text, "what time is it")
        self.assertGreater(heard.audio_seconds, 0.7)
        self.assertGreaterEqual(heard.latency_s, 0.0)
        self.assertEqual(rec.transcribing, [LaneState.ACTIVE, LaneState.DONE])

    async def test_text_without_words_is_dropped(self):
        rec = Recording(["..."])
        listener = self.make(rec)
        self.feed_all(listener, frames())
        await listener.settled()
        self.assertEqual(rec.heard, [])
        self.assertEqual(rec.transcribing, [LaneState.ACTIVE, LaneState.DONE])

    async def test_recognition_errors_are_reported_not_raised(self):
        rec = Recording()
        rec.fail = True
        listener = self.make(rec)
        with self.assertLogs("voicemate.pipeline.listener", "ERROR"):
            self.feed_all(listener, frames())
            await listener.settled()
        self.assertEqual(rec.heard, [])
        self.assertEqual([str(e) for e in rec.errors], ["ASR crashed"])
        self.assertEqual(rec.transcribing, [LaneState.ACTIVE, LaneState.IDLE])

    async def test_audio_is_saved_when_a_folder_is_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recording()
            listener = self.make(rec, audio_dir=Path(tmp) / "audio")
            self.feed_all(listener, frames())
            await listener.settled()
            self.assertEqual(len(list((Path(tmp) / "audio").glob("*.wav"))), 1)

    async def test_nothing_is_saved_by_default(self):
        rec = Recording()
        listener = self.make(rec)
        self.feed_all(listener, frames())
        await listener.settled()
        self.assertEqual(len(rec.heard), 1)


class TestPartials(ListenerTestCase):
    async def test_partials_arrive_while_the_user_is_still_speaking(self):
        rec = Recording(["partial words"])
        listener = self.make(rec, partial_interval_s=0.5)
        for frame in frames(speech_s=2.0, silence_s=0.0):
            listener.feed(frame)
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        self.assertGreaterEqual(len(rec.partials), 1)
        self.assertEqual(rec.partials[0], "partial words")
        self.assertEqual(rec.heard, [])  # still speaking: no final utterance yet

    async def test_a_long_utterance_only_transcribes_the_recent_audio(self):
        rec = Recording(["words"])
        listener = self.make(rec, partial_interval_s=0.5, partial_window_s=0.6)
        for frame in frames(speech_s=2.0, silence_s=0.0):
            listener.feed(frame)
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        self.assertGreaterEqual(len(rec.partials), 2)
        self.assertFalse(rec.partials[0].startswith("… "))  # still shorter than the window
        self.assertTrue(rec.partials[-1].startswith("… "))  # later ones cover only the end
        self.assertLessEqual(max(rec.seen_audio), int(0.6 * SAMPLE_RATE) + 1)
        self.assertEqual(PARTIAL_WINDOW_S, 8.0)

    async def test_a_late_partial_is_dropped_once_speech_has_ended(self):
        rec = Recording(["late"])
        rec.gate = asyncio.Event()
        listener = self.make(rec, partial_interval_s=0.5)
        pcm = frames(speech_s=1.0, silence_s=0.9)
        for frame in pcm:
            listener.feed(frame)
            await asyncio.sleep(0)
        rec.gate.set()
        await listener.settled()
        await asyncio.sleep(0.05)
        self.assertEqual(rec.partials, [])


class TestFlushSettleStop(ListenerTestCase):
    async def test_flush_ends_the_sentence_in_progress(self):
        rec = Recording(["are you there"])
        listener = self.make(rec)
        for frame in frames()[:40]:
            listener.feed(frame)
        self.assertTrue(listener.in_speech)
        end = listener.flush()
        self.assertIsInstance(end, SpeechEnd)
        self.assertFalse(listener.in_speech)
        await listener.settled()
        self.assertEqual([h.text for h in rec.heard], ["are you there"])

    async def test_flush_in_silence_does_nothing(self):
        rec = Recording()
        listener = self.make(rec)
        self.assertIsNone(listener.flush())
        await listener.settled()
        self.assertEqual(rec.heard, [])

    async def test_settled_waits_for_the_recognizer_and_the_callback(self):
        rec = Recording(["slow one"])
        rec.gate = asyncio.Event()
        rec.slow_utterance = True
        listener = self.make(rec)
        self.feed_all(listener, frames())
        waiting = asyncio.create_task(listener.settled())
        await asyncio.sleep(0.05)
        self.assertFalse(waiting.done())
        rec.gate.set()
        await waiting
        self.assertEqual([h.text for h in rec.heard], ["slow one"])  # callback finished too

    async def test_strict_mode_is_forwarded_to_the_detector(self):
        listener = self.make(Recording())
        self.assertFalse(listener.strict)
        listener.strict = True
        self.assertTrue(listener.segmenter.strict)


if __name__ == "__main__":
    unittest.main()
