# Lifecycle stage 10 - Model Deployment

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data.preprocess import clean_text
from src.quarantine.store import list_quarantined, quarantine_email


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "src" / "web"
MODEL_PATH = PROJECT_ROOT / "models" / "spam_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "count_vectorizer.pkl"

app = FastAPI(title="SignalShield API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


class Email(BaseModel):
    text: str = Field(..., min_length=3, max_length=100_000)


def classify(text: str) -> dict:
    features = vectorizer.transform([clean_text(text)])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    scores = {label: round(float(probabilities[index]), 4) for index, label in enumerate(classes)}
    label = str(model.predict(features)[0])
    confidence = scores.get(label, 0.0)
    return {"label": label, "confidence": confidence, "scores": scores}


@app.get("/health")
def health() -> dict:
    return {"status": "operational", "model": MODEL_PATH.name, "version": app.version}


@app.get("/api/overview")
def overview() -> dict:
    records = list_quarantined(limit=1000)
    spam_count = sum(record.get("label") == "spam" for record in records)
    return {
        "scans": len(records),
        "threats": spam_count,
        "protected": max(len(records) - spam_count, 0),
        "quarantine": len(records),
        "detection_rate": round((spam_count / len(records)) * 100, 1) if records else 0,
    }


@app.get("/api/quarantine")
def quarantine(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    return {"items": list_quarantined(limit)}


@app.post("/api/predict")
@app.post("/predict")
def predict(email: Email) -> dict:
    result = classify(email.text)
    if result["label"] == "spam":
        quarantine_email(email.text, result["label"], result["confidence"])
    return result


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend is not installed")
    return FileResponse(index)
