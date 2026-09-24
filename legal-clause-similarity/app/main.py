# ============================================================
# app/main.py
# ============================================================

from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException
)

from fastapi.responses import (
    FileResponse
)

from fastapi.staticfiles import (
    StaticFiles
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import BaseModel, Field

from app.search_engine import (
    search_similar_clauses
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


FRONTEND_DIR = (
    BASE_DIR
    / "frontend"
)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Legal Clause Similarity Engine",

    description=(
        "Semantic legal clause retrieval "
        "using TF-IDF Weighted Word2Vec."
    ),

    version="1.0.0"

)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)

app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class SearchRequest(
    BaseModel
):

    query: str = Field(

        ...,

        min_length=1,

        description=(
            "Legal clause, sentence, "
            "or paragraph."
        )

    )

    top_k: int = Field(

        default=5,

        ge=1,

        le=20

    )

    clause_type: str = Field(

        default="all",

        description="Optional clause type filter."

    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def home():

    return FileResponse(

        FRONTEND_DIR
        / "index.html"

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "service":
            "Legal Clause Similarity Engine"

    }


# ============================================================
# SEARCH API
# ============================================================

@app.post(
    "/search"
)
def search(
    request: SearchRequest
):

    try:

        results = (
            search_similar_clauses(

                query=request.query,

                top_k=request.top_k,

                clause_type=request.clause_type

            )
        )


        return {

            "success":
                True,

            "query":
                request.query,

            "top_k":
                request.top_k,

            "results":
                results

        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )