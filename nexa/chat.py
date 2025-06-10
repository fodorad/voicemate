import sounddevice as sd
import numpy as np
import tempfile
import time
from pathlib import Path

# Import custom modules
from nexa import STT_DIR, TTS_DIR, LLM_DIR  
from nexa.STT.wav2vec2_hu import Wav2Vec2
from nexa.TTS.mms_hu import MMS
from nexa.LLM.llama_hu import SambaLingo

"""
  0 ASUS VP249, Core Audio (0 in, 2 out)
> 1 Lorgar Rapax microphone, Core Audio (2 in, 0 out)
< 2 Mac mini Speakers, Core Audio (0 in, 2 out)
  3 Microsoft Teams Audio, Core Audio (1 in, 1 out)
"""
sd.default.device = (1, 2)
prompt_context = "kontextus: te egy chatbot vagy, a neved Nexa. A felhasználó kérdéseket tehet fel. Célod a nagyon rövid és lényegretörő választ adni. Ne használj komplex nyelvet vagy hosszú magyarázatot. A számokat szövegben kell írnod. A hang zajos vagy hibás lehet, próbáld kitalálni mit mondhatott. \n\n Kérdés:"


class VoiceChatbot:
    def __init__(self):
        # Initialize modules
        print("Betöltöm a modulokat...")
        self.stt = Wav2Vec2()
        self.tts = MMS()
        self.llm = SambaLingo()
        self.sample_rate = 16000  # Sample rate for audio recording
        self.chunk_duration = 5   # Record for 5 seconds at a time
        
    def record_audio(self, duration=5):
        """Record audio from the microphone"""
        print(f"\nHallgatom a beszédet ({duration} másodpercig)...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        return audio_data
    
    def save_temp_audio(self, audio_data):
        """Save audio data to a temporary WAV file"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, self.sample_rate)
            return temp_file.name

    def save_audio(self, audio_data, file_path):
        """Save audio data to a WAV file"""
        sf.write(file_path, audio_data, self.sample_rate)
    
    def save_text(self, text, file_path):
        """Save text to a file"""
        with open(file_path, 'w') as f:
            f.write(text)
    
    def run(self):
        try:
            # Initial greeting
            print("\nSzia, figyelek!")
            greeting_audio = self.tts.synthesize("Szia, figyelek!")
            sd.play(greeting_audio, self.sample_rate)
            sd.wait()
            
            while True:
                # Record audio
                audio_data = self.record_audio(self.chunk_duration)
                
                # Save to temp file for STT
                temp_audio_path = self.save_temp_audio(audio_data)
                
                try:
                    # Speech to Text
                    print("Feldolgozom a beszédet...")
                    prompt = self.stt.transcribe(Path(temp_audio_path))
                    self.save_audio(audio_data, STT_DIR / "test_hu.wav")
                    print("Kontextus:", prompt_context)
                    print("Te:", prompt)
                    
                    # Generate response with LLM
                    print("Válasz generálása...")
                    response = self.llm.generate(prompt_context + prompt)
                    self.save_text(response, LLM_DIR / "test_hu.txt")
                    print(f"Bot: {response}")
                    
                    # Text to Speech
                    print("Hangátalakítás...")
                    audio = self.tts.synthesize(response)
                    self.save_audio(audio, TTS_DIR / "test_hu.wav")
                    sd.play(audio, self.sample_rate)
                    sd.wait()
                    
                except Exception as e:
                    print(f"Hiba történt: {str(e)}")
                    error_audio = self.tts.synthesize("Elnézést, hiba történt. Próbálja újra.")
                    sd.play(error_audio, self.sample_rate)
                    sd.wait()
                
                finally:
                    # Clean up temp file
                    Path(temp_audio_path).unlink(missing_ok=True)
                    
        except KeyboardInterrupt:
            print("\nViszlát!")
            goodbye_audio = self.tts.synthesize("Viszlát!")
            sd.play(goodbye_audio, self.sample_rate)
            sd.wait()

if __name__ == "__main__":
    import soundfile as sf  # Import here to avoid circular imports
    chatbot = VoiceChatbot()
    chatbot.run()
