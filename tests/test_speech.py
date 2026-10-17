import asyncio
import threading
import time
import unittest

import numpy as np

from tests.helpers import ScriptedRecognizer, ToneSynthesizer
from voicemate.speech import Speech
from voicemate.tts.voices import VoiceBank


class Gauge:
    """Records how many calls overlap and which thread ran them."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.threads: set[str] = set()
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
            self.threads.add(threading.current_thread().name)
        time.sleep(0.03)
        with self._lock:
            self.running -= 1


class SlowRecognizer:
    def __init__(self, gauge: Gauge) -> None:
        self.gauge = gauge

    def transcribe_text(self, audio: np.ndarray) -> str:
        self.gauge.enter()
        return "heard"


class SlowVoice(ToneSynthesizer):
    def __init__(self, gauge: Gauge) -> None:
        super().__init__(16000)
        self.gauge = gauge

    def synthesize(self, text: str) -> np.ndarray:
        self.gauge.enter()
        return super().synthesize(text)


def audio() -> np.ndarray:
    return np.zeros(1600, dtype=np.float32)


class TestSpeech(unittest.IsolatedAsyncioTestCase):
    def make(self, recognizer=None, voice=None) -> Speech:
        voices = VoiceBank({"en": voice or ToneSynthesizer(22050)}, default_lang="en")
        speech = Speech(recognizer or ScriptedRecognizer(["hello there"]), voices)
        self.addAsyncCleanup(speech.aclose)
        return speech

    async def test_transcribe_returns_the_recognizers_text(self):
        self.assertEqual(await self.make().transcribe(audio()), "hello there")

    async def test_synthesize_routes_by_language_and_returns_the_rate(self):
        pcm, rate = await self.make().synthesize("Hello", "en")
        self.assertEqual(rate, 22050)
        self.assertEqual(pcm.dtype, np.float32)
        self.assertGreater(len(pcm), 0)

    async def test_unspeakable_text_gives_empty_audio(self):
        pcm, rate = await self.make().synthesize("https://x.y/a", "en")
        self.assertEqual((len(pcm), rate), (0, 22050))

    async def test_models_run_off_the_loop_one_call_at_a_time(self):
        gauge = Gauge()
        speech = self.make(SlowRecognizer(gauge), SlowVoice(gauge))
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.create_task(ticker())
        await asyncio.gather(*(speech.transcribe(audio()) for _ in range(4)))
        task.cancel()
        self.assertGreater(ticks, 5)  # the loop stayed responsive while models worked
        self.assertEqual(gauge.peak, 1)
        self.assertTrue(all(name.startswith("asr") for name in gauge.threads))

        gauge.threads.clear()
        await asyncio.gather(*(speech.synthesize("hi", "en") for _ in range(4)))
        self.assertEqual(gauge.peak, 1)
        self.assertTrue(all(name.startswith("tts") for name in gauge.threads))

    async def test_asr_and_tts_do_not_block_each_other(self):
        gauge = Gauge()
        speech = self.make(SlowRecognizer(gauge), SlowVoice(gauge))
        await asyncio.gather(speech.transcribe(audio()), speech.synthesize("hi", "en"))
        self.assertEqual(gauge.peak, 2)

    async def test_use_after_close_fails_clearly(self):
        speech = self.make()
        await speech.aclose()
        await speech.aclose()  # closing twice is harmless
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await speech.transcribe(audio())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await speech.synthesize("hi", "en")

    async def test_recognizer_errors_reach_the_caller(self):
        class Broken:
            def transcribe_text(self, audio):
                raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            await self.make(Broken()).transcribe(audio())


if __name__ == "__main__":
    unittest.main()
