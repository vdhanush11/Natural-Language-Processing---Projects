from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import MODEL_DIR
from backend.app.schemas.prediction import (
    ExplanationRequest,
    ExplanationResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.model_service import FakeNewsModel, ModelNotReadyError

app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description=(
        "FastAPI backend for the DistilBERT fake/real news classifier "
        "exported from the supplied Colab notebook."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None
startup_error = None

try:
    detector = FakeNewsModel()
except Exception as exc:
    startup_error = str(exc)


@app.get("/api/health")
def health():
    return {
        "status": "ok" if detector else "model_not_ready",
        "model_loaded": detector is not None,
        "model_directory": str(MODEL_DIR),
        "error": startup_error,
    }


@app.get("/api/model-info")
def model_info():
    if detector is None:
        raise HTTPException(status_code=503, detail=startup_error)

    return {
        "model": detector.model_name,
        "task": "binary text classification",
        "labels": detector.label_map,
        "max_length": 256,
        "device": str(detector.device),
        "model_directory": str(MODEL_DIR),
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail=startup_error)

    try:
        return detector.predict(request.title, request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/explain", response_model=ExplanationResponse)
def explain(request: ExplanationRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail=startup_error)

    try:
        return detector.explain(request.title, request.text, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
