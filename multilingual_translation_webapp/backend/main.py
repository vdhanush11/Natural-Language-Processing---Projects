from pathlib import Path
from typing import Optional, List
import os
import time
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .translator import TranslationService, LANGUAGES, SUPPORTED_PAIRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Lexora Multilingual Translation API",
    description="FastAPI inference service for the NLLB-200 distilled 600M multilingual Transformer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = TranslationService()

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=12000)
    target_language: str
    source_language: Optional[str] = None
    protect_terms: bool = True
    num_beams: int = Field(5, ge=1, le=8)
    max_new_tokens: int = Field(512, ge=32, le=1024)
    no_repeat_ngram_size: int = Field(3, ge=0, le=8)
    length_penalty: float = Field(1.0, ge=0.1, le=3.0)

class BatchRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1, max_length=100)
    target_language: str
    source_language: Optional[str] = None
    protect_terms: bool = True

@app.get("/api/health")
def health():
    return service.health()

@app.get("/api/languages")
def languages():
    return {
        "languages": LANGUAGES,
        "supported_pairs": [list(p) for p in SUPPORTED_PAIRS],
    }

@app.get("/api/model")
def model_info():
    return service.model_info()

@app.post("/api/translate")
def translate(req: TranslationRequest):
    try:
        return service.translate(
            text=req.text,
            tgt_iso=req.target_language,
            src_iso=req.source_language,
            protect=req.protect_terms,
            generation={
                "num_beams": req.num_beams,
                "max_new_tokens": req.max_new_tokens,
                "no_repeat_ngram_size": req.no_repeat_ngram_size,
                "length_penalty": req.length_penalty,
                "early_stopping": True,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logging.exception("Translation failed")
        raise HTTPException(status_code=500, detail=f"Translation failed: {type(exc).__name__}")

@app.post("/api/translate/batch")
def translate_batch(req: BatchRequest):
    try:
        return {
            "results": service.translate_batch(
                messages=req.messages,
                tgt_iso=req.target_language,
                src_iso=req.source_language,
                protect=req.protect_terms,
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logging.exception("Batch translation failed")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {type(exc).__name__}")

@app.post("/api/translate/file")
async def translate_file(
    file: UploadFile = File(...),
    target_language: str = "hi",
    source_language: Optional[str] = None,
):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Upload must be UTF-8 text or CSV.")

    # One message per non-empty line. CSV support is intentionally lightweight:
    # the first column is treated as the message when the file has comma-separated rows.
    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            line = line.split(",", 1)[0].strip().strip('"')
        messages.append(line)

    messages = messages[:100]
    if not messages:
        raise HTTPException(status_code=400, detail="No translatable rows found.")

    try:
        return {
            "filename": file.filename,
            "count": len(messages),
            "results": service.translate_batch(messages, target_language, source_language),
        }
    except Exception as exc:
        logging.exception("File translation failed")
        raise HTTPException(status_code=500, detail=f"File translation failed: {type(exc).__name__}")

# Serve the polished frontend from the same FastAPI process.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
