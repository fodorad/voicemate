
from pathlib import Path
from abc import ABC, abstractmethod


class SpeechToTextAbstract(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        pass

    @abstractmethod
    def pipeline(self, audio: bytes, file_path: Path):
        pass