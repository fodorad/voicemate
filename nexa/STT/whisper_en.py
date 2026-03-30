from pathlib import Path
import torch
import whisper
from nexa import TTS_DIR, STT_DIR
from nexa.STT import SpeechToTextAbstract


class Whisper(SpeechToTextAbstract):

    def __init__(self, model_size: str = "base"):
        super().__init__()
        self.model = whisper.load_model(model_size)
        self.device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
        self.model = self.model.to(self.device)

    def transcribe(self, audio: Path) -> str:
        result = self.model.transcribe(str(audio))
        return result["text"]

    def pipeline(self, audio: Path, file_path: Path):
        text = self.transcribe(audio)
        with open(file_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    audio = TTS_DIR / "test_en.wav"
    stt = Whisper()
    stt.pipeline(audio, STT_DIR / "test_whisper_en.txt")