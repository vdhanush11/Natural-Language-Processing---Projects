# Lifecycle stage 8 — Model Evaluation
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parents[2]
input_path = base_dir / "data" / "processed" / "cleaned_reviews.csv"
model_path = base_dir / "models" / "sentiment_model.pkl"
vectorizer_path = base_dir / "models" / "tfidf_vectorizer.pkl"

df = pd.read_csv(input_path)

if "clean_review" not in df.columns or "sentiment" not in df.columns:
    raise KeyError("The processed dataset is missing required columns for evaluation.")

mask = df["clean_review"].notna() & df["clean_review"].astype(str).str.strip().ne("")
df = df.loc[mask].copy()
df["clean_review"] = df["clean_review"].fillna("").astype(str)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

X = vectorizer.transform(df["clean_review"])
y = df["sentiment"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_test, predictions, labels=labels)
cm_table = pd.DataFrame(
    cm,
    index=["actual_negative", "actual_neutral", "actual_positive"],
    columns=["pred_negative", "pred_neutral", "pred_positive"],
)

print("\nConfusion Matrix (rows = actual, columns = predicted)")
print(cm_table)
