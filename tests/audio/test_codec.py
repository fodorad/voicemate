import unittest

import numpy as np

from voicemate.audio.codec import (
    float32_to_pcm16,
    pack_audio_frame,
    pcm16_to_float32,
    resample,
    unpack_audio_frame,
)


class TestPCMConversion(unittest.TestCase):
    def test_roundtrip_preserves_shape_and_values(self):
        audio = np.random.uniform(-1, 1, 1600).astype(np.float32)
        pcm = float32_to_pcm16(audio)
        self.assertEqual(len(pcm), 3200)
        back = pcm16_to_float32(pcm)
        self.assertEqual(back.dtype, np.float32)
        self.assertEqual(back.shape, (1600,))
        np.testing.assert_allclose(back, audio, atol=1 / 32767 + 1e-6)

    def test_out_of_range_samples_are_clipped(self):
        pcm = float32_to_pcm16(np.array([2.0, -2.0], dtype=np.float32))
        np.testing.assert_array_equal(np.frombuffer(pcm, dtype="<i2"), [32767, -32767])

    def test_odd_byte_count_is_rejected(self):
        with self.assertRaises(ValueError):
            pcm16_to_float32(b"\x00\x01\x02")


class TestResample(unittest.TestCase):
    def test_output_length_follows_rate_ratio(self):
        audio = np.random.randn(48000).astype(np.float32)
        out = resample(audio, 48000, 16000)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (16000,))

    def test_same_rate_returns_input(self):
        audio = np.random.randn(100).astype(np.float32)
        self.assertIs(resample(audio, 16000, 16000), audio)


class TestFraming(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        pcm = float32_to_pcm16(np.random.uniform(-1, 1, 480).astype(np.float32))
        sample_rate, payload = unpack_audio_frame(pack_audio_frame(pcm, 24000))
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(payload, pcm)

    def test_short_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            unpack_audio_frame(b"\x01\x02")


if __name__ == "__main__":
    unittest.main()
