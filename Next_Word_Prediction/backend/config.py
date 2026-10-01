
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CORPUS_PATH = PROJECT_ROOT / "reference" / "Articles.csv"

SEED = 42
SEQUENCE_LENGTH = 10
VOCABULARY_SIZE = 10_000
TOP_K_DEFAULT = 5
MAX_TOP_K = 10
MAX_GENERATION_WORDS = 30
GPT_MAX_INPUT_TOKENS = 128

RNN_FILES = {
    "Simple RNN": MODEL_DIR / "simplernn_best.keras",
    "LSTM": MODEL_DIR / "lstm_best.keras",
    "GRU": MODEL_DIR / "gru_best.keras",
}
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
SEQUENCE_CONFIG_PATH = MODEL_DIR / "sequence_config.json"
GPT_DIR = MODEL_DIR / "distilgpt2"

APP_HOST = os.getenv("NWP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("NWP_PORT", "8000"))
