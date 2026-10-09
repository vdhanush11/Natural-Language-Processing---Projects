from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "fake_news_model"

MAX_LEN = 256
DEFAULT_TOP_K = 6
ALLOWED_ORIGINS = ["*"]
