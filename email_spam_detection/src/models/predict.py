# Lifecycle stage 7 — Model Testing
 
import joblib
from src.data.preprocess import clean_text
 
model = joblib.load("models/spam_model.pkl")
 
vectorizer = joblib.load("models/count_vectorizer.pkl")
 
while True:
 
    email = input("\nEnter Email: ")
 
    X = vectorizer.transform([clean_text(email)])
 
    prediction = model.predict(X)
 
    print("\nPrediction:", prediction[0])
