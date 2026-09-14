# Lifecycle stage 4 — Data Preparation (feature extraction)
 
import pandas as pd
import joblib
 
from sklearn.feature_extraction.text import TfidfVectorizer
 
 
df = pd.read_csv("data/processed/cleaned_news.csv")
df["clean_text"] = df["clean_text"].fillna("")
 
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
 
X = vectorizer.fit_transform(df["clean_text"])
 
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
 
print(X.shape)
