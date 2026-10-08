from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import torch
from docx import Document
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "bge-large-en-v1.5"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
LOCAL_MODEL_PATH = Path(os.getenv("RESUME_SCREENING_LOCAL_MODEL", str(DEFAULT_LOCAL_MODEL_PATH)))
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_WORDS = 60

SECTION_HEADINGS = {
    "summary": [
        "summary", "professional summary", "career summary", "profile",
        "about me", "objective", "career objective"
    ],
    "skills": [
        "skills", "technical skills", "key skills", "core competencies",
        "technical expertise", "technologies", "tech stack", "skill set"
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "career history", "employment"
    ],
    "projects": [
        "projects", "academic projects", "personal projects",
        "key projects", "project experience"
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "educational qualification", "qualifications"
    ],
    "certifications": [
        "certifications", "certification", "courses", "training",
        "licenses and certifications"
    ],
    "achievements": [
        "achievements", "awards", "accomplishments", "honors",
        "publications", "extracurricular"
    ],
}

JD_HEADING_MAP = {
    "role": [
        "job title", "title", "role", "position", "designation",
        "about the role", "role overview", "job summary",
        "about the job", "overview"
    ],
    "required_skills": [
        "required skills", "requirements", "must have", "must haves",
        "required qualifications", "skills required", "key skills",
        "technical skills", "what we are looking for", "who you are",
        "essential skills", "mandatory skills", "technical requirements"
    ],
    "preferred_skills": [
        "preferred skills", "nice to have", "good to have", "bonus",
        "preferred qualifications", "desirable", "plus points",
        "added advantage"
    ],
    "responsibilities": [
        "responsibilities", "key responsibilities", "what you will do",
        "what you'll do", "duties", "job description", "your role",
        "day to day"
    ],
    "experience": [
        "experience", "experience required", "work experience",
        "years of experience", "eligibility"
    ],
    "education": [
        "education", "qualification", "qualifications",
        "educational qualification", "academic requirements"
    ],
    "company": [
        "about us", "about the company", "who we are", "company overview"
    ],
    "benefits": [
        "benefits", "what we offer", "perks", "compensation", "salary"
    ],
}

DROP_SECTIONS = {"company", "benefits"}

SECTION_MAP = {
    "required_skills": ["skills", "experience", "projects"],
    "experience": ["experience", "summary"],
    "responsibilities": ["experience", "projects"],
    "preferred_skills": ["skills", "projects", "experience"],
}

SECTION_WEIGHTS = {
    "required_skills": 0.40,
    "experience": 0.25,
    "responsibilities": 0.20,
    "preferred_skills": 0.15,
}

SKILL_ALIASES = {
    "python": ["python"],
    "sql": ["sql"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "pyspark": ["pyspark"],
    "apache spark": ["apache spark", "spark"],
    "airflow": ["airflow", "apache airflow"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git"],
    "linux": ["linux"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "nlp": ["nlp", "natural language processing"],
    "transformers": ["transformers", "transformer"],
    "bert": ["bert"],
    "rest api": ["rest api", "restful api", "rest apis"],
    "kafka": ["kafka", "apache kafka"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft excel"],
    "data warehousing": ["data warehouse", "data warehousing"],
    "etl": ["etl", "extract transform load", "etl/elt", "elt"],
    "data modeling": ["data modeling", "data modelling", "dimensional modeling", "dimensional modelling"],
}

_model: SentenceTransformer | None = None


@dataclass
class ResumeInput:
    filename: str
    content: bytes


def get_model() -> SentenceTransformer:
    global _model

    if _model is not None:
        return _model

    if not LOCAL_MODEL_PATH.exists() or not LOCAL_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Local BGE model was not found at: {LOCAL_MODEL_PATH}\n"
            "Copy your complete 'bge-large-en-v1.5' folder into the project's "
            "'models' directory, or set RESUME_SCREENING_LOCAL_MODEL to its location."
        )

    required_files = [
        LOCAL_MODEL_PATH / "model.safetensors",
        LOCAL_MODEL_PATH / "modules.json",
        LOCAL_MODEL_PATH / "config_sentence_transformers.json",
        LOCAL_MODEL_PATH / "1_Pooling",
        LOCAL_MODEL_PATH / "2_Normalize",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The BGE model folder is incomplete. Missing:\n" + "\n".join(missing)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = SentenceTransformer(str(LOCAL_MODEL_PATH), device=device)
    return _model


def extract_text(content: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        document = fitz.open(stream=content, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in document)
        finally:
            document.close()

    if suffix == ".docx":
        document = Document(io.BytesIO(content))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return "\n".join(parts)

    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}. Use PDF, DOCX, or TXT.")


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("\t", " ")
    text = re.sub(r"[•‣▪◦●·∙]", "-", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#./\-,()&@:\n ]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def match_heading(line: str) -> str | None:
    line = line.strip().strip(":").strip("-").strip()

    if len(line.split()) > 5 or len(line) < 3:
        return None

    for section, variants in SECTION_HEADINGS.items():
        for variant in variants:
            if line == variant or line.startswith(variant):
                return section

    return None


def split_into_sections(text: str) -> tuple[dict[str, str], bool]:
    sections: dict[str, str] = {}
    current = "header"
    buffer: list[str] = []

    for line in text.split("\n"):
        heading = match_heading(line)

        if heading:
            if buffer:
                sections[current] = sections.get(current, "") + "\n" + "\n".join(buffer)
            current = heading
            buffer = []
        elif line.strip():
            buffer.append(line.strip())

    if buffer:
        sections[current] = sections.get(current, "") + "\n" + "\n".join(buffer)

    sections = {key: value.strip() for key, value in sections.items() if value.strip()}
    real_sections = [key for key in sections if key != "header"]

    if len(real_sections) < 2:
        return {"full_text": text.strip()}, True

    return sections, False


def clean_heading_candidate(line: str) -> str:
    value = line.strip()
    value = value.strip("#").strip()
    value = value.replace("*", "").replace("_", "")
    value = value.strip("-").strip("=").strip()
    value = value.strip(":").strip()
    return value.lower().strip()


def lookup_jd_section(candidate: str) -> str | None:
    if len(candidate) < 3 or len(candidate.split()) > 6:
        return None

    for section, variants in JD_HEADING_MAP.items():
        for variant in variants:
            if candidate == variant or candidate.startswith(variant):
                return section

    return None


def match_jd_heading(line: str) -> tuple[str | None, str]:
    raw = line.strip()

    if not raw or set(raw) <= set("-=_#* "):
        return None, ""

    head, separator, rest = raw.partition(":")

    if separator:
        section = lookup_jd_section(clean_heading_candidate(head))
        if section:
            return section, rest.strip()

    section = lookup_jd_section(clean_heading_candidate(raw))
    if section:
        return section, ""

    return None, ""


def split_jd_into_sections(text: str) -> tuple[dict[str, str], bool]:
    sections: dict[str, str] = {}
    current = "role"
    buffer: list[str] = []

    def flush():
        if buffer:
            existing = sections.get(current, "")
            sections[current] = (existing + "\n" + "\n".join(buffer)).strip()

    for line in text.split("\n"):
        section, inline = match_jd_heading(line)

        if section:
            flush()
            current = section
            buffer = [inline] if inline else []
        elif line.strip() and not set(line.strip()) <= set("-=_#* "):
            buffer.append(line.strip())

    flush()

    sections = {key: value.strip() for key, value in sections.items() if value.strip()}

    for dropped in DROP_SECTIONS:
        sections.pop(dropped, None)

    if len([key for key in sections if key != "role"]) < 1:
        return {"full_text": text.strip()}, True

    return sections, False


def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    count = 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        words = line.split()
        n_words = len(words)

        if count + n_words > max_words and buffer:
            chunks.append(" ".join(buffer))
            buffer = []
            count = 0

        if n_words > max_words:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                count = 0

            for start in range(0, n_words, max_words):
                chunks.append(" ".join(words[start:start + max_words]))
        else:
            buffer.append(line)
            count += n_words

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    if not texts:
        dimension = get_model().get_sentence_embedding_dimension()
        return np.zeros((0, dimension), dtype="float32")

    if is_query:
        texts = [QUERY_PREFIX + text for text in texts]

    return get_model().encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_sections(sections: dict[str, str], is_query: bool = False):
    chunks = {section: chunk_text(text) for section, text in sections.items()}
    vectors = {section: embed(parts, is_query=is_query) for section, parts in chunks.items()}
    return chunks, vectors


def compare_section(
    jd_vectors: np.ndarray,
    jd_chunks: list[str],
    resume_vectors: dict[str, np.ndarray],
    resume_chunks: dict[str, list[str]],
    targets: list[str],
):
    if "full_text" in resume_vectors:
        targets = ["full_text"]

    matrices = []
    evidence_texts = []
    evidence_sources = []

    for target in targets:
        if target in resume_vectors and len(resume_vectors[target]) > 0:
            matrices.append(resume_vectors[target])
            evidence_texts.extend(resume_chunks[target])
            evidence_sources.extend([target] * len(resume_chunks[target]))

    if not matrices or len(jd_vectors) == 0:
        return None, []

    resume_matrix = np.vstack(matrices)
    similarity = jd_vectors @ resume_matrix.T

    best_indices = similarity.argmax(axis=1)
    best_values = similarity.max(axis=1)
    score = float(best_values.mean())

    evidence = []
    for index, value in enumerate(best_values):
        match_index = int(best_indices[index])
        evidence.append({
            "jd_chunk": jd_chunks[index],
            "resume_chunk": evidence_texts[match_index],
            "resume_section": evidence_sources[match_index],
            "sim": round(float(value), 4),
        })

    evidence.sort(key=lambda item: item["sim"], reverse=True)
    return score, evidence[:3]


def extract_years(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
    ]

    values: list[float] = []
    for pattern in patterns:
        values.extend(float(value) for value in re.findall(pattern, text.lower()))

    return max(values) if values else 0.0


def remove_personal_details(text: str) -> str:
    cleaned = str(text)
    patterns = [
        r"(?im)^\s*(?:name|full name)\s*[:\-].*$",
        r"(?im)^\s*(?:age|date of birth|dob)\s*[:\-].*$",
        r"(?im)^\s*(?:gender|sex)\s*[:\-].*$",
        r"(?im)^\s*(?:marital status|religion|nationality)\s*[:\-].*$",
        r"(?im)\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        r"(?im)\b(?:linkedin\.com/in|github\.com/)[^\s]+\b",
        r"(?im)\b(?:\+?\d[\d\s().-]{8,}\d)\b",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)
    lines = cleaned.splitlines()
    if lines:
        lines = lines[1:] if len(lines) > 1 else lines
    return "\n".join(lines).strip()


def extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    found = []

    for canonical, aliases in SKILL_ALIASES.items():
        if any(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", lowered) for alias in aliases):
            found.append(canonical)

    return sorted(found)


def display_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.title()


def build_reason(section_scores: dict[str, float | None], matched_skills: list[str], years: float) -> str:
    available = [(key, value) for key, value in section_scores.items() if value is not None]
    available.sort(key=lambda pair: pair[1], reverse=True)

    if not available:
        return "Semantic similarity could not be established from the structured JD sections."

    strongest = available[0][0].replace("_", " ")

    if matched_skills:
        skill_text = ", ".join(matched_skills[:5])
        return (
            f"Strongest semantic alignment is in {strongest}; matched skills include "
            f"{skill_text}. Detected experience: {years:g} years."
        )

    return (
        f"Strongest semantic alignment is in {strongest}. "
        f"Detected experience: {years:g} years."
    )


def calculate_raw_score(section_scores: dict[str, float | None]) -> float:
    weighted_total = 0.0
    used_weight = 0.0

    for section, weight in SECTION_WEIGHTS.items():
        value = section_scores.get(section)

        if value is not None and np.isfinite(value):
            weighted_total += float(value) * weight
            used_weight += weight

    if used_weight > 0:
        return weighted_total / used_weight

    values = [
        float(value)
        for value in section_scores.values()
        if value is not None and np.isfinite(value)
    ]

    return float(np.mean(values)) if values else float("nan")


def screen_resumes(
    jd_content: bytes,
    jd_filename: str,
    resumes: list[ResumeInput],
    top_k: int,
) -> dict[str, Any]:
    if not resumes:
        raise ValueError("Upload at least one resume.")

    if top_k < 1:
        raise ValueError("Shortlist size must be at least 1.")

    jd_raw = extract_text(jd_content, jd_filename).strip()

    if not jd_raw:
        raise ValueError("The job description is empty or could not be parsed.")

    cleaned_jd = preprocess(jd_raw)
    jd_sections, jd_fallback = split_jd_into_sections(cleaned_jd)
    jd_chunks, jd_vectors = embed_sections(jd_sections, is_query=True)

    results = []

    for resume in resumes:
        raw_text = extract_text(resume.content, resume.filename).strip()

        if not raw_text:
            continue

        clean_text = preprocess(raw_text)
        resume_sections, used_fallback = split_into_sections(clean_text)

        if used_fallback:
            ranking_text = remove_personal_details(clean_text)
            resume_sections = {"full_text": ranking_text}

        resume_chunks, resume_vectors = embed_sections(resume_sections, is_query=False)

        section_scores: dict[str, float | None] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}

        for jd_section, targets in SECTION_MAP.items():
            if jd_section not in jd_vectors:
                continue

            score, section_evidence = compare_section(
                jd_vectors[jd_section],
                jd_chunks[jd_section],
                resume_vectors,
                resume_chunks,
                targets,
            )

            section_scores[jd_section] = score
            evidence[jd_section] = section_evidence

        raw_score = calculate_raw_score(section_scores)
        years = extract_years(raw_text)
        skills = extract_skills(raw_text)

        results.append({
            "file_name": resume.filename,
            "name": display_name(resume.filename),
            "raw_score": raw_score,
            "years": years,
            "skills": skills,
            "section_scores": section_scores,
            "evidence": evidence,
            "reason": build_reason(section_scores, skills, years),
            "word_count": len(raw_text.split()),
            "used_fallback": used_fallback,
        })

    if not results:
        raise ValueError("No readable resumes were found.")

    scores = np.array([item["raw_score"] for item in results], dtype=float)
    valid_mask = np.isfinite(scores)

    if not valid_mask.any():
        raise ValueError("No valid semantic similarity scores were produced.")

    median = float(np.nanmedian(scores))
    scores[~valid_mask] = median

    low = float(scores.min())
    high = float(scores.max())

    for item, score in zip(results, scores):
        item["raw_score"] = float(score)
        item["fit_score"] = 50.0 if high - low < 1e-9 else round((score - low) / (high - low) * 100, 1)

    results.sort(key=lambda item: item["raw_score"], reverse=True)

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return {
        "model": MODEL_NAME,
        "model_path": str(LOCAL_MODEL_PATH),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "jd_file": jd_filename,
        "jd_word_count": len(jd_raw.split()),
        "jd_sections": sorted(jd_sections.keys()),
        "jd_fallback": jd_fallback,
        "candidate_count": len(results),
        "top_k": min(top_k, len(results)),
        "shortlist": results[:top_k],
        "all_results": results,
    }
