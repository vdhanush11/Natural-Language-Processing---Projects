# Lifecycle stage 5 — Model Building (TF-IDF features)
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

base_dir = Path(__file__).resolve().parents[2]
csv_path = base_dir / "data" / "processed" / "cleaned_reviews.csv"
model_path = base_dir / "models" / "tfidf_vectorizer.pkl"

df = pd.read_csv(csv_path)

if "clean_review" not in df.columns:
    raise KeyError("The processed dataset does not contain the 'clean_review' column.")

texts = df["clean_review"].fillna("").astype(str)
texts = texts[texts.str.strip() != ""]

df = df.loc[texts.index].copy()
df["clean_review"] = texts

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_review"])

model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(vectorizer, model_path)

print(X.shape)
