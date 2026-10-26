import http.server
import io
import json
import os
import threading
import unittest

import numpy as np
import soundfile

from voicemate.tts.phonon import PHONON_SAMPLE_RATE, PhononError, PhononSynthesizer


def _wav(seconds: float = 0.2, rate: int = PHONON_SAMPLE_RATE) -> bytes:
    samples = np.sin(np.linspace(0, 200, int(seconds * rate))).astype(np.float32) * 0.3
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class _Ollama(http.server.BaseHTTPRequestHandler):
    """A tiny stand-in for the Ollama endpoints Phonon uses."""

    version = "0.40.11"
    speech_status = 200
    speech_body = b""
    requests: list[dict] = []

    def _reply(self, status: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._reply(200, json.dumps({"version": self.version}).encode())

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).requests.append(json.loads(self.rfile.read(length)))
        self._reply(self.speech_status, self.speech_body, "audio/wav")

    def log_message(self, *args):
        pass


class TestPhononSynthesizer(unittest.TestCase):
    def serve(self, **attributes) -> str:
        handler = type("Handler", (_Ollama,), {"requests": [], **attributes})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.handler = handler
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_text_becomes_mono_float_audio_at_the_engine_rate(self):
        voice = PhononSynthesizer(host=self.serve(speech_body=_wav(0.2)))
        audio = voice.synthesize("Hello there.")
        self.assertEqual(voice.sample_rate, PHONON_SAMPLE_RATE)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (int(0.2 * PHONON_SAMPLE_RATE),))
        request = self.handler.requests[0]
        self.assertEqual(
            (request["model"], request["voice"], request["input"]),
            ("phonon", "default", "Hello there."),
        )

    def test_blank_text_makes_no_request(self):
        voice = PhononSynthesizer(host=self.serve(speech_body=_wav()))
        self.assertEqual(len(voice.synthesize("  ")), 0)
        self.assertEqual(self.handler.requests, [])

    def test_old_ollama_is_refused_with_the_needed_version(self):
        with self.assertRaisesRegex(PhononError, "0.40.9"):
            PhononSynthesizer(host=self.serve(version="0.40.0"))

    def test_missing_model_says_how_to_pull_it(self):
        voice = PhononSynthesizer(host=self.serve(speech_status=404, speech_body=b"no model"))
        with self.assertRaisesRegex(PhononError, "ollama pull phonon"):
            voice.synthesize("Hello.")

    def test_other_server_errors_are_reported(self):
        voice = PhononSynthesizer(host=self.serve(speech_status=500, speech_body=b"boom"))
        with self.assertRaisesRegex(PhononError, "500"):
            voice.synthesize("Hello.")

    def test_wrong_sample_rate_is_rejected(self):
        voice = PhononSynthesizer(host=self.serve(speech_body=_wav(rate=16000)))
        with self.assertRaisesRegex(PhononError, "16000"):
            voice.synthesize("Hello.")

    def test_unreachable_ollama_is_reported(self):
        with self.assertRaisesRegex(PhononError, "not reachable"):
            PhononSynthesizer(host="http://127.0.0.1:9", timeout_s=1.0)


@unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION"), "needs Ollama with phonon")
class TestRealPhonon(unittest.TestCase):  # pragma: no cover - integration only
    def test_speech_is_audible_and_fast(self):
        import time

        voice = PhononSynthesizer()
        started = time.perf_counter()
        audio = voice.synthesize("Best counters are Muradin and Johanna.")
        elapsed = time.perf_counter() - started
        self.assertGreater(len(audio) / voice.sample_rate, 1.0)
        self.assertGreater(float(np.abs(audio).max()), 0.05)
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
