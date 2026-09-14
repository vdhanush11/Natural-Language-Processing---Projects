# Lifecycle stage 8 — Model Evaluation
 
import pandas as pd
import joblib
 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
 
 
df = pd.read_csv("data/processed/cleaned_news.csv")
df["clean_text"] = df["clean_text"].fillna("")
 
model = joblib.load("models/news_model.pkl")
 
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
 
X = vectorizer.transform(df["clean_text"])
y = df["news_category"]
 
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
predictions = model.predict(X_test)
 
print(classification_report(y_test, predictions))
 
# List the categories once so the matrix rows/columns are labeled.
labels = sorted(y.unique())
 
cm = confusion_matrix(y_test, predictions, labels=labels)
 
cm_table = pd.DataFrame(cm, index=labels, columns=labels)
 
print("\nConfusion Matrix (rows = actual, columns = predicted)")
print(cm_table)
