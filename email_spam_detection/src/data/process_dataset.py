# Lifecycle stage 4 — Data Preparation (full dataset)
 
import pandas as pd
from src.data.preprocess import clean_batch
from src.data.load_data import load_data
 
def process_dataset():
 
    print("Script Started")
 
    df = load_data("data/raw/email_spam.csv")
 
    print("Dataset Loaded")
 
    # Lowercasing + regex are fast as vectorized pandas ops,
    # so we do them here and leave only tokenizing/lemmatizing to spaCy.
 
    texts = df["message"].astype(str).str.lower()
    texts = texts.str.replace(r"http\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z ]", "", regex=True)
 
    df["clean_email"] = clean_batch(texts)
 
    print("Cleaning Completed")
 
    df.to_csv("data/processed/cleaned_emails.csv", index=False)
 
    print("File Saved")
