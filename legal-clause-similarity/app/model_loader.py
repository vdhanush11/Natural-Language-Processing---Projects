# ============================================================
# app/model_loader.py
# ============================================================

from pathlib import Path
import pickle

from gensim.models import Word2Vec


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# MODEL FILE PATHS
# ============================================================

WORD2VEC_PATH = (
    BASE_DIR
    / "models"
    / "word2vec.model"
)

CLAUSE_VECTORS_PATH = (
    BASE_DIR
    / "models"
    / "clause_vectors.pkl"
)

TFIDF_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.pkl"
)

LEGAL_DOCS_PATH = (
    BASE_DIR
    / "data"
    / "legal_docs.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    WORD2VEC_PATH,
    CLAUSE_VECTORS_PATH,
    TFIDF_PATH,
    LEGAL_DOCS_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# LOAD WORD2VEC
# ============================================================

word2vec_model = Word2Vec.load(
    str(WORD2VEC_PATH)
)


# ============================================================
# LOAD CLAUSE VECTORS
# ============================================================

with open(
    CLAUSE_VECTORS_PATH,
    "rb"
) as file:

    clause_vectors = pickle.load(
        file
    )


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

with open(
    TFIDF_PATH,
    "rb"
) as file:

    tfidf_vectorizer = pickle.load(
        file
    )


# ============================================================
# LOAD LEGAL DOCUMENTS
# ============================================================

import pandas as pd

legal_docs = pd.read_csv(
    LEGAL_DOCS_PATH
)


# ============================================================
# VALIDATE DATA
# ============================================================

if len(clause_vectors) != len(legal_docs):

    raise ValueError(
        "Mismatch between clause vectors "
        "and legal documents."
    )


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("=" * 70)

print(
    "LEGAL CLAUSE SIMILARITY ENGINE"
)

print("=" * 70)

print(
    f"Number of clauses: {len(legal_docs)}"
)

print(
    f"Word2Vec vocabulary: "
    f"{len(word2vec_model.wv.index_to_key)}"
)

print(
    f"Word2Vec vector size: "
    f"{word2vec_model.vector_size}"
)

print(
    f"Clause vector shape: "
    f"{clause_vectors.shape}"
)

print(
    f"TF-IDF vocabulary: "
    f"{len(tfidf_vectorizer.get_feature_names_out())}"
)

print("=" * 70)