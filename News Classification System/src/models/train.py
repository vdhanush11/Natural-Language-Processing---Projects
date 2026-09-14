# Lifecycle stage 6 — Model Training
 
import pandas as pd
import joblib
from collections import Counter
 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.models.build_model import build_model
 
 
df = pd.read_csv("data/processed/cleaned_news.csv")
df["clean_text"] = df["clean_text"].fillna("")
 
# Load the vectorizer prepared in Stage 4 — transform only, never re-fit.
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
 
X = vectorizer.transform(df["clean_text"])
y = df["news_category"]
 
# Split FIRST — so balancing can never touch the test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
# Balance the TRAINING split only: duplicate rows from the smaller
# categories until every category has as many as the largest one.
print("Before balancing:", Counter(y_train))
 
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)
 
print("After balancing: ", Counter(y_train))
 
# Build the untrained model from Stage 5, then fit it on the balanced data.
model = build_model()
 
model.fit(X_train, y_train)
 
joblib.dump(model, "models/news_model.pkl")
 
print("Training Completed")
