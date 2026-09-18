# Lifecycle stage 6 — Model Training
 
import pandas as pd
import joblib
 
from sklearn.model_selection import train_test_split

if __package__:
    from src.models.build_model import build_model
else:
    from build_model import build_model
 
 
df = pd.read_csv("data/processed/cleaned_emails.csv")
df["clean_email"] = df["clean_email"].fillna("")
 
# Load the vectorizer prepared in Stage 4 — transform only, never re-fit.
vectorizer = joblib.load("models/count_vectorizer.pkl")
 
X = vectorizer.transform(df["clean_email"])
y = df["label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
# Build the untrained model from Stage 5, then fit it here.
model = build_model()
 
model.fit(X_train, y_train)
 
joblib.dump(model, "models/spam_model.pkl")
 
print("Training Completed")
