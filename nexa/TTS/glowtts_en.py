from TTS.api import TTS
import torch
import tempfile
from pathlib import Path
from nexa import TTS_DIR
from nexa.TTS import TextToSpeechAbstract


class GlowTTS(TextToSpeechAbstract):

    def __init__(self):
        super().__init__()
        self.tts = TTS("tts_models/en/ljspeech/glow-tts")

        self.device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
        self.model = self.model.to(self.device)

    def synthesize(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            self.tts.tts_to_file(text=text, file_path=temp_file.name)
            return temp_file.read()

    def pipeline(self, text: str, file_path: Path):
        self.tts.tts_to_file(text=text, file_path=str(file_path))


if __name__ == "__main__":
    tts = GlowTTS()
    tts.pipeline("Hi, how can I help you?", TTS_DIR / "test_en.wav")
