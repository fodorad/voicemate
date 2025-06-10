from abc import ABC, abstractmethod
from pathlib import Path


class TextToSpeechAbstract(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        pass

    @abstractmethod
    def pipeline(self, text: str, file_path: Path):
        pass
        