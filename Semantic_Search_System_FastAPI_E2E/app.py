from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from backend.config import AVAILABLE_MODELS, DEFAULT_TOP_K, MAX_TOP_K
from backend.search_engine import SearchEngine
from backend.evaluation import load_evaluation_metrics

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Semantic Search System",
    description="Semantic document retrieval using Word2Vec, FastText and TF-IDF.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
engine = SearchEngine()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str = Field(default="FastText")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"status": engine.get_status(),
                 "metrics": load_evaluation_metrics()},
    )


@app.get("/api/status")
async def status():
    return engine.get_status()


@app.get("/api/evaluation")
async def evaluation():
    return load_evaluation_metrics()


@app.post("/api/search")
async def search(payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(400, "Please enter a search query.")
    if payload.model not in AVAILABLE_MODELS:
        raise HTTPException(
            400, f"Model must be one of: {', '.join(AVAILABLE_MODELS)}"
        )
    try:
        return engine.search(query, payload.model, payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/compare")
async def compare(payload: CompareRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(400, "Please enter a search query.")
    try:
        return engine.compare_models(query, payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
