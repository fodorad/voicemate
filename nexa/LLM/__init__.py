from abc import ABC, abstractmethod
from pathlib import Path


class LargeLanguageModelAbstract(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(self, text: str) -> str:
        pass

    @abstractmethod
    def pipeline(self, text: str, file_path: Path):
        pass