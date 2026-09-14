# Lifecycle stage 9 - Model Deployment

from __future__ import annotations

import csv
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.models.prediction_service import PredictionService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parent / "static"
FEED_DIR = PROJECT_ROOT / "data" / "feeds"


class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10_000)
    route: bool = True
    include_cleaned_text: bool = False
    top_k: int = Field(default=3, ge=1, le=7)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-space character")
        return value.strip()


class BatchRequest(BaseModel):
    articles: list[ArticleRequest] = Field(..., min_length=1, max_length=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService.load()
    yield


app = FastAPI(
    title="NewsLens Classification API",
    description="Production API for classifying and routing news articles.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def get_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return service


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict[str, str | bool]:
    service = getattr(request.app.state, "prediction_service", None)
    return {
        "status": "ok" if service else "degraded",
        "model_loaded": service is not None,
    }


@app.get("/api/categories")
def categories(request: Request) -> dict[str, list[str]]:
    return {"categories": get_service(request).categories}


@app.post("/api/predict")
def predict(article: ArticleRequest, request: Request) -> dict:
    return get_service(request).predict(
        article.text,
        route=article.route,
        include_cleaned_text=article.include_cleaned_text,
        top_k=article.top_k,
    )


@app.post("/api/predict/batch")
def predict_batch(payload: BatchRequest, request: Request) -> dict:
    service = get_service(request)
    results = [
        service.predict(
            article.text,
            route=article.route,
            include_cleaned_text=article.include_cleaned_text,
            top_k=article.top_k,
        )
        for article in payload.articles
    ]
    return {"count": len(results), "results": results}


@app.get("/api/feeds")
def feeds(
    request: Request,
    category: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    get_service(request)
    files = [FEED_DIR / f"{category}.csv"] if category else sorted(FEED_DIR.glob("*.csv"))
    articles: list[dict[str, str]] = []
    for path in files:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as feed_file:
            articles.extend(csv.DictReader(feed_file))
    articles.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return {"count": len(articles[:limit]), "articles": articles[:limit]}


@app.get("/api/stats")
def stats(request: Request) -> dict:
    service = get_service(request)
    counts = {category: 0 for category in service.categories}
    for path in FEED_DIR.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as feed_file:
            counts[path.stem] = sum(1 for _ in csv.DictReader(feed_file))
    return {
        "categories": len(service.categories),
        "category_counts": counts,
        "total_routed": sum(counts.values()),
    }
