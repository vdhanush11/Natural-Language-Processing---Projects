import pandas as pd
from .config import MODEL_COMPARISON

def load_evaluation_metrics():
    if not MODEL_COMPARISON.exists():
        return {
            "available": False,
            "message": "Copy results/model_comparison.csv from Colab.",
            "rows": [],
        }
    try:
        df = pd.read_csv(MODEL_COMPARISON)
        return {
            "available": True,
            "message": "Evaluation metrics loaded.",
            "rows": df.fillna("").to_dict(orient="records"),
        }
    except Exception as exc:
        return {
            "available": False,
            "message": f"Could not load evaluation metrics: {exc}",
            "rows": [],
        }
