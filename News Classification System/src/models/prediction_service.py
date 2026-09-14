from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.preprocess import clean_text
from src.routing.store import route_article


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "news_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


@dataclass
class PredictionService:
    model: Any
    vectorizer: Any

    @classmethod
    def load(cls) -> "PredictionService":
        missing = [
            str(path)
            for path in (MODEL_PATH, VECTORIZER_PATH)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Model artifacts are missing: " + ", ".join(missing)
            )

        return cls(
            model=joblib.load(MODEL_PATH),
            vectorizer=joblib.load(VECTORIZER_PATH),
        )

    @property
    def categories(self) -> list[str]:
        return sorted(str(category) for category in self.model.classes_)

    def predict(
        self,
        text: str,
        *,
        route: bool = True,
        include_cleaned_text: bool = False,
        top_k: int = 3,
    ) -> dict[str, Any]:
        cleaned_text = clean_text(text)
        features = self.vectorizer.transform([cleaned_text])
        scores = self.model.decision_function(features)

        if np.ndim(scores) == 1:
            scores = np.asarray(scores).reshape(1, -1)
        probabilities = self._softmax(scores[0])
        ranked_indexes = np.argsort(probabilities)[::-1][:top_k]
        predictions = [
            {
                "category": str(self.model.classes_[index]),
                "confidence": round(float(probabilities[index]), 4),
            }
            for index in ranked_indexes
        ]
        category = predictions[0]["category"]

        if route:
            route_article(text, category)

        result: dict[str, Any] = {
            "category": category,
            "confidence": predictions[0]["confidence"],
            "alternatives": predictions,
        }
        if include_cleaned_text:
            result["cleaned_text"] = cleaned_text
        return result

    def predict_batch(
        self,
        texts: list[str],
        *,
        route: bool = True,
        include_cleaned_text: bool = False,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return [
            self.predict(
                text,
                route=route,
                include_cleaned_text=include_cleaned_text,
                top_k=top_k,
            )
            for text in texts
        ]

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        values = np.exp(shifted)
        return values / values.sum()
