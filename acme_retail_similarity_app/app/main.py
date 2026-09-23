from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR
from .model_service import service
from .schemas import CompareRequest, RecommendationRequest, TrainRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load catalogue and any existing artifacts.
    # TF-IDF is trained automatically if no artifact exists.
    service.initialize()
    yield


app = FastAPI(
    title="Acme Retail Product Similarity API",
    version="1.0.0",
    description=(
        "Production-style product recommendation API based on the "
        "supplied TF-IDF, Word2Vec and FastText notebook workflow."
    ),
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "acme-retail-product-similarity",
    }


@app.get("/api/status")
def status():
    return service.stats()


@app.get("/api/categories")
def categories():
    return service.categories()


@app.get("/api/products")
def products(
    search: str | None = Query(default=None),
    main_category: str | None = Query(default=None),
    sub_category: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
):
    return {
        "products": service.products(
            search=search,
            main_category=main_category,
            sub_category=sub_category,
            limit=limit,
        )
    }


@app.post("/api/recommend")
def recommend(payload: RecommendationRequest):
    try:
        return service.recommend(
            query_text=payload.query_text,
            catalog_id=payload.catalog_id,
            model=payload.model,
            top_n=payload.top_n,
            main_category=payload.main_category,
            sub_category=payload.sub_category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {exc}",
        ) from exc


@app.post("/api/compare")
def compare(payload: CompareRequest):
    try:
        return service.compare(
            query_text=payload.query_text,
            catalog_id=payload.catalog_id,
            top_n=payload.top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {exc}",
        ) from exc


@app.post("/api/train")
def train(payload: TrainRequest):
    try:
        service.train(payload.model)
        return {
            "status": "completed",
            "model": payload.model,
            "models": service.status,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {exc}",
        ) from exc
