# Lifecycle stage 4 — Data Preparation (full dataset)
from pathlib import Path

import pandas as pd

from src.data.preprocess import clean_batch, create_sentiment


def process_dataset():
    print("Script Started")

    base_dir = Path(__file__).resolve().parents[2]
    csv_path = base_dir / "data" / "raw" / "Amazon Product Reviews Dataset.csv"
    output_path = base_dir / "data" / "processed" / "cleaned_reviews.csv"

    df = pd.read_csv(csv_path)
    print("Dataset Loaded")

    if "Text" not in df.columns:
        if "reviews.text" in df.columns:
            df = df.rename(columns={"reviews.text": "Text"})
        else:
            raise KeyError("The dataset does not contain a usable review text column.")

    sample_size = min(50000, len(df))
    df = df.sample(sample_size, random_state=42)

    df["sentiment"] = df["Text"].fillna("").apply(create_sentiment)
    print("Sentiment Created")

    texts = df["Text"].astype(str).str.lower()
    texts = texts.str.replace(r"http\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z ]", "", regex=True)

    df["clean_review"] = clean_batch(texts)
    print("Cleaning Completed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("File Saved")
