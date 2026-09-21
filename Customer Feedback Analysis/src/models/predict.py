# Lifecycle stage 7 — Model Testing
from pathlib import Path
import sys

import joblib

base_dir = Path(__file__).resolve().parents[2]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.data.preprocess import clean_text, create_sentiment

model = joblib.load(base_dir / "models" / "sentiment_model.pkl")
vectorizer = joblib.load(base_dir / "models" / "tfidf_vectorizer.pkl")

while True:
    review = input("\nEnter Review: ")
    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])
    model_prediction = model.predict(X)[0]
    rule_prediction = create_sentiment(review)

    if model_prediction != rule_prediction:
        prediction = rule_prediction
    else:
        prediction = model_prediction

    print("\nPrediction:", prediction)