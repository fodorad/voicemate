import warnings
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
from nexa import TTS_DIR, STT_DIR
from nexa.STT import SpeechToTextAbstract


warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")


class Wav2Vec2(SpeechToTextAbstract):

    def __init__(self):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
        self.model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")

        self.device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
        self.model = self.model.to(self.device)

    def transcribe(self, audio: Path) -> str:
        audio, sr = sf.read(audio)
        if sr != 16000:
            raise ValueError("A mintavételezésnek 16kHz-nek kell lennie!")

        input_values = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            input_values = input_values.to(self.device)
            logits = self.model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1).squeeze()
        transcription = self.processor.decode(predicted_ids, skip_special_tokens=True)
        
        return transcription

    def pipeline(self, audio: Path, file_path: Path):
        text = self.transcribe(audio)
        with open(file_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    audio = TTS_DIR / "test_hu.wav"
    stt = Wav2Vec2()
    stt.pipeline(audio, STT_DIR / "test_hu.txt")