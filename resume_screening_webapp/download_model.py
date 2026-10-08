from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "bge-large-en-v1.5"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model = SentenceTransformer(MODEL_NAME)
model.save(str(MODEL_PATH))
print(f"Model saved to: {MODEL_PATH}")
