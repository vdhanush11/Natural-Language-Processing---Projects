
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, SEED, SEQUENCE_LENGTH, VOCABULARY_SIZE
from .schemas import PredictionRequest, PredictionResponse, GenerateRequest, GenerateResponse
from .model_service import service

app = FastAPI(
    title="Next Word Prediction API",
    version="1.0.0",
    description="Production-style inference API for Simple RNN, LSTM, GRU and DistilGPT-2."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    statuses = service.model_status()
    return {
        "status": "ok",
        "models": statuses,
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "vocabulary_size": VOCABULARY_SIZE,
    }

@app.get("/api/models")
def models():
    return {
        "models": service.model_status(),
        "loadable_models": service.loadable_models(),
    }

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return service.predict(request.text, request.model, request.top_k)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        return service.generate(request.text, request.model, request.num_words)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

@app.get("/api/evaluation")
def evaluation():
    import json
    path = Path(__file__).resolve().parents[1] / "models" / "phase_10_model_comparison.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation file not found.")
    return json.loads(path.read_text(encoding="utf-8"))

# Serve frontend after API routes.
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
