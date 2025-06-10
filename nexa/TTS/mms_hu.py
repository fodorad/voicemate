from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
from pathlib import Path
from nexa import TTS_DIR
from nexa.TTS import TextToSpeechAbstract


class MMS(TextToSpeechAbstract):

    def __init__(self):
        super().__init__()
        self.model = VitsModel.from_pretrained("facebook/mms-tts-hun")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hun")

        self.device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
        self.model = self.model.to(self.device)

    def synthesize(self, text: str) -> bytes:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform

        output = output.cpu().squeeze().numpy()
        return output

    def pipeline(self, text: str, file_path: Path):
        output = self.synthesize(text)
        sf.write(str(file_path), output, 16000)


if __name__ == "__main__":
    tts = MMS()
    tts.pipeline("Szia, miben segíthetek?", TTS_DIR / "test_hu.wav")
