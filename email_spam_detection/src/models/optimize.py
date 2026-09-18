# Lifecycle stage 9 — Model Optimization (handling imbalance)
 
import pandas as pd
import joblib
from collections import Counter
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
 
if __package__:
    from src.models.build_model import build_model
else:
    from build_model import build_model
 
 
df = pd.read_csv("data/processed/cleaned_emails.csv")
df["clean_email"] = df["clean_email"].fillna("")
 
# Keep all ham, but only 25 spam -> a deliberately imbalanced set
ham = df[df["label"] == "ham"]
spam = df[df["label"] == "spam"].head(25)
 
imbalanced = pd.concat([ham, spam])
 
print("Class counts:", Counter(imbalanced["label"]))
vectorizer = joblib.load("models/count_vectorizer.pkl")
 
X = vectorizer.transform(imbalanced["clean_email"])
y = imbalanced["label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
baseline = build_model()
baseline.fit(X_train, y_train)
 
print("--- Baseline (imbalanced) ---")
print(classification_report(y_test, baseline.predict(X_test)))

print("Before:", Counter(y_train))
 
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
 
print("After: ", Counter(y_res))
 
optimized = build_model()
optimized.fit(X_res, y_res)
 
print("--- After oversampling ---")
print(classification_report(y_test, optimized.predict(X_test)))
