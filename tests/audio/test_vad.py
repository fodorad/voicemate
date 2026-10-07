import os
import unittest

import numpy as np

from voicemate.audio.codec import SAMPLE_RATE
from voicemate.audio.vad import (
    EnergyVAD,
    Segmenter,
    SileroVAD,
    SpeechEnd,
    SpeechStart,
)
from voicemate.config import VADConfig


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def feed_in_chunks(segmenter: Segmenter, audio: np.ndarray, chunk: int = 320) -> list:
    events = []
    for start in range(0, len(audio), chunk):
        events.extend(segmenter.feed(audio[start : start + chunk]))
    return events


class TestEnergyVAD(unittest.TestCase):
    def test_probability_separates_tone_from_silence(self):
        vad = EnergyVAD()
        self.assertGreater(vad(tone(0.032)[: vad.frame_size]), 0.9)
        self.assertLess(vad(silence(0.032)[: vad.frame_size]), 0.1)


class TestSegmenter(unittest.TestCase):
    def setUp(self):
        self.config = VADConfig(min_silence_ms=300, min_speech_ms=200, preroll_ms=100)
        self.segmenter = Segmenter(EnergyVAD(), self.config)

    def test_one_utterance_yields_start_then_end_with_audio(self):
        audio = np.concatenate([silence(0.5), tone(1.0), silence(0.6)])
        events = feed_in_chunks(self.segmenter, audio)
        self.assertEqual([type(e) for e in events], [SpeechStart, SpeechEnd])
        utterance = events[1].audio
        # Speech plus pre-roll, trailing silence trimmed to a short tail.
        self.assertGreater(len(utterance), int(1.0 * SAMPLE_RATE))
        self.assertLess(len(utterance), int(1.5 * SAMPLE_RATE))
        self.assertFalse(self.segmenter.in_speech)

    def test_short_blip_is_ignored(self):
        audio = np.concatenate([silence(0.3), tone(0.1), silence(0.8)])
        self.assertEqual(feed_in_chunks(self.segmenter, audio), [])

    def test_short_pause_does_not_split_utterance(self):
        audio = np.concatenate([tone(0.6), silence(0.15), tone(0.6), silence(0.6)])
        events = feed_in_chunks(self.segmenter, audio)
        self.assertEqual([type(e) for e in events], [SpeechStart, SpeechEnd])

    def test_two_utterances(self):
        audio = np.concatenate([tone(0.5), silence(0.6), tone(0.5), silence(0.6)])
        kinds = [type(e) for e in feed_in_chunks(self.segmenter, audio)]
        self.assertEqual(kinds, [SpeechStart, SpeechEnd, SpeechStart, SpeechEnd])

    def test_current_audio_grows_during_speech(self):
        feed_in_chunks(self.segmenter, tone(0.5))
        self.assertTrue(self.segmenter.in_speech)
        self.assertGreater(len(self.segmenter.current_audio), int(0.4 * SAMPLE_RATE))
        self.assertGreater(self.segmenter.speech_seconds, 0.4)

    def test_strict_mode_requires_longer_speech(self):
        self.segmenter.strict = True  # barge_in_min_speech_ms=300 by default
        self.assertEqual(
            feed_in_chunks(self.segmenter, np.concatenate([tone(0.25), silence(0.6)])), []
        )
        events = feed_in_chunks(self.segmenter, np.concatenate([tone(0.5), silence(0.6)]))
        self.assertIsInstance(events[0], SpeechStart)

    def test_max_utterance_forces_end(self):
        segmenter = Segmenter(EnergyVAD(), self.config, max_utterance_s=1.0)
        kinds = [type(e) for e in feed_in_chunks(segmenter, tone(1.5))]
        self.assertEqual(kinds[:2], [SpeechStart, SpeechEnd])

    def test_flush_ends_the_utterance_in_progress(self):
        feed_in_chunks(self.segmenter, tone(0.5))
        end = self.segmenter.flush()
        self.assertIsInstance(end, SpeechEnd)
        self.assertGreater(len(end.audio), int(0.4 * SAMPLE_RATE))
        self.assertFalse(self.segmenter.in_speech)
        self.assertIsNone(self.segmenter.flush())  # nothing left

    def test_flush_drops_a_blip_too_short_to_be_speech(self):
        feed_in_chunks(self.segmenter, tone(0.1))
        self.assertIsNone(self.segmenter.flush())
        self.assertEqual(feed_in_chunks(self.segmenter, silence(0.5)), [])

    def test_reset_discards_partial_utterance(self):
        feed_in_chunks(self.segmenter, tone(0.5))
        self.segmenter.reset()
        self.assertFalse(self.segmenter.in_speech)
        self.assertEqual(len(self.segmenter.current_audio), 0)


class TestSileroVAD(unittest.TestCase):
    """Runs the real bundled Silero ONNX model (small, ships with the silero-vad wheel)."""

    def test_frame_probability_is_in_unit_interval(self):
        vad = SileroVAD()
        self.assertEqual(vad.frame_size, 512)
        for frame in (silence(0.032), np.random.uniform(-0.5, 0.5, 512).astype(np.float32)):
            prob = vad(frame[: vad.frame_size])
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        self.assertLess(vad(silence(0.032)[:512]), 0.2)
        vad.reset()

    def test_wrong_frame_size_is_rejected(self):
        with self.assertRaises(ValueError):
            SileroVAD()(silence(0.01))

    @unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION"), "integration only")
    def test_detects_synthetic_speech(self):  # pragma: no cover - needs macOS `say`
        import subprocess
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(
                ["say", "-o", tmp.name, "--data-format=LEI16@16000", "Hello, this is a test."],
                check=True,
            )
            audio, _ = sf.read(tmp.name, dtype="float32")
        segmenter = Segmenter(SileroVAD(), VADConfig())
        events = feed_in_chunks(segmenter, np.concatenate([silence(0.5), audio, silence(1.0)]))
        self.assertIsInstance(events[0], SpeechStart)
        self.assertIsInstance(events[-1], SpeechEnd)


if __name__ == "__main__":
    unittest.main()
