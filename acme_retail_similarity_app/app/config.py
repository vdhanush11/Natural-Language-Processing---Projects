from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv(
    "ACME_DATA_PATH",
    str(BASE_DIR / "data" / "AcmeRetail_Products_70000.csv")
))
MODEL_DIR = Path(os.getenv(
    "ACME_MODEL_DIR",
    str(BASE_DIR / "models")
))
STATIC_DIR = BASE_DIR / "app" / "static"

TEXT_COLUMNS = ["name", "main_category", "sub_category"]
OUTPUT_COLUMNS = [
    "catalog_id",
    "name",
    "main_category",
    "sub_category",
    "ratings",
    "no_of_ratings",
    "discount_price",
    "actual_price",
    "image",
    "link",
    "similarity_score",
]

TFIDF_MAX_FEATURES = 100_000
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = int(os.getenv("W2V_EPOCHS", "10"))
W2V_WORKERS = int(os.getenv("W2V_WORKERS", "4"))
FASTTEXT_VECTOR_SIZE = 100
FASTTEXT_WINDOW = 5
FASTTEXT_MIN_COUNT = 2
FASTTEXT_EPOCHS = int(os.getenv("FASTTEXT_EPOCHS", "10"))
FASTTEXT_WORKERS = int(os.getenv("FASTTEXT_WORKERS", "4"))

TOP_N_DEFAULT = 10
TOP_N_MAX = 50
