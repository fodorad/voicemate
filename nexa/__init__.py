from pathlib import Path


PROJECT_DIR = Path(__file__).parents[1]
DATA_DIR = PROJECT_DIR / "data"
STT_DIR = DATA_DIR / "STT"
TTS_DIR = DATA_DIR / "TTS"
LLM_DIR = DATA_DIR / "LLM"