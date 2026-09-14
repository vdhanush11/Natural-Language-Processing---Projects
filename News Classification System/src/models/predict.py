# Lifecycle stage 7 — Model Testing
 
import joblib
from src.data.preprocess import clean_text
 
model = joblib.load("models/news_model.pkl")
 
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
 
while True:
 
    text = input("\nEnter News: ")
 
    X = vectorizer.transform([clean_text(text)])
 
    prediction = model.predict(X)
 
    print("\nPrediction:", prediction[0])
