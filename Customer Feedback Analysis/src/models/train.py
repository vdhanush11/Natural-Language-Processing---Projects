# Lifecycle stage 6 — Model Training
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parents[2]
input_path = base_dir / "data" / "processed" / "cleaned_reviews.csv"
model_path = base_dir / "models" / "sentiment_model.pkl"
vectorizer_path = base_dir / "models" / "tfidf_vectorizer.pkl"

df = pd.read_csv(input_path)

if "clean_review" not in df.columns or "sentiment" not in df.columns:
    raise KeyError("The processed dataset is missing required columns for model training.")

valid_mask = df["clean_review"].notna() & df["clean_review"].astype(str).str.strip().ne("")
df = df.loc[valid_mask].copy()
df["clean_review"] = df["clean_review"].fillna("").astype(str)

X = df["clean_review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("Training Completed")
