from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "nllb-200-distilled-600M"
MODEL = "facebook/nllb-200-distilled-600M"

TARGET.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL} into {TARGET}")
AutoTokenizer.from_pretrained(MODEL).save_pretrained(TARGET)
AutoModelForSeq2SeqLM.from_pretrained(MODEL).save_pretrained(TARGET)
print("Model download complete.")
print(f"Set MODEL_DIR={TARGET}")
