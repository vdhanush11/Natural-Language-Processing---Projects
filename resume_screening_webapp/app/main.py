from __future__ import annotations

import csv
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import ResumeInput, MODEL_NAME, screen_resumes


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Resume Screening AI",
    version="1.0.0",
    description="Semantic resume screening using FastAPI and BGE sentence embeddings."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "Resume Screening AI",
        "model": MODEL_NAME,
    }


@app.post("/api/screen")
async def screen(
    jd_file: UploadFile = File(...),
    resumes: list[UploadFile] = File(...),
    top_k: int = Form(5),
):
    try:
        jd_content = await jd_file.read()

        resume_inputs = []
        for upload in resumes:
            content = await upload.read()
            resume_inputs.append(
                ResumeInput(
                    filename=upload.filename or "resume",
                    content=content,
                )
            )

        result = screen_resumes(
            jd_content=jd_content,
            jd_filename=jd_file.filename or "job_description",
            resumes=resume_inputs,
            top_k=top_k,
        )

        return JSONResponse(result)

    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )


@app.post("/api/export-csv")
async def export_csv(payload: dict):
    rows = payload.get("results", [])

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "rank",
            "name",
            "file_name",
            "fit_score",
            "raw_score",
            "years",
            "reason",
        ],
    )
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "rank": row.get("rank"),
            "name": row.get("name"),
            "file_name": row.get("file_name"),
            "fit_score": row.get("fit_score"),
            "raw_score": row.get("raw_score"),
            "years": row.get("years"),
            "reason": row.get("reason"),
        })

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=resume_screening_results.csv"
        },
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
