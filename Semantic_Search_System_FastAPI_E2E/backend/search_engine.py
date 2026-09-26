import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import (
    PROCESSED_TEST_FILE, TEST_WORD2VEC_DOCUMENTS,
    TEST_FASTTEXT_DOCUMENTS, AVAILABLE_MODELS, DEFAULT_TOP_K
)
from .models import ArtifactStore
from .preprocessing import preprocess_text

class SearchEngine:
    def __init__(self):
        self.available_models = AVAILABLE_MODELS
        self.store = ArtifactStore()
        self.documents = None
        self.word2vec_documents = None
        self.fasttext_documents = None
        self.tfidf_documents = None
        self._load_document_artifacts()

    def _load_document_artifacts(self):
        if not self.store.loaded:
            return
        if not PROCESSED_TEST_FILE.exists():
            self.store.error = f"Missing {PROCESSED_TEST_FILE.name} in outputs/."
            return
        try:
            self.documents = pd.read_csv(PROCESSED_TEST_FILE)
            if TEST_WORD2VEC_DOCUMENTS.exists():
                self.word2vec_documents = np.load(TEST_WORD2VEC_DOCUMENTS)
            if TEST_FASTTEXT_DOCUMENTS.exists():
                self.fasttext_documents = np.load(TEST_FASTTEXT_DOCUMENTS)
            self.tfidf_documents = self.store.tfidf_vectorizer.transform(
                self.documents["search_text"].fillna("").astype(str)
            )
        except Exception as exc:
            self.store.error = f"Could not load document artifacts: {exc}"

    @property
    def ready(self):
        return (
            self.store.loaded and self.documents is not None
            and self.word2vec_documents is not None
            and self.fasttext_documents is not None
            and self.tfidf_documents is not None
        )

    def get_status(self):
        return {
            "ready": self.ready,
            "models": self.available_models,
            "documents": 0 if self.documents is None else len(self.documents),
            "embedding_dimension": self.store.embedding_dim,
            "message": "Search engine ready." if self.ready else (
                self.store.error or "Search engine artifacts are not ready."
            ),
        }

    @staticmethod
    def _safe_cosine(query_vector, matrix):
        qnorm = np.linalg.norm(query_vector)
        if qnorm == 0:
            return np.zeros(len(matrix), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        denom = qnorm * norms
        scores = np.zeros(len(matrix), dtype=np.float32)
        valid = denom > 0
        if np.any(valid):
            scores[valid] = (matrix[valid] @ query_vector) / denom[valid]
        return scores

    def _semantic_query_vector(self, query, model):
        vectors = []
        for word in preprocess_text(query):
            if model == "Word2Vec":
                vector = self.store.get_word2vec_vector(word)
                if vector is not None:
                    vectors.append(vector)
            else:
                vector = self.store.get_fasttext_vector(word)
                if np.linalg.norm(vector) > 0:
                    vectors.append(vector)
        if not vectors:
            return np.zeros(self.store.embedding_dim, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def _rank_results(self, scores, top_k):
        indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        results = []
        for rank, idx in enumerate(indices, 1):
            row = self.documents.iloc[int(idx)]
            results.append({
                "rank": rank,
                "document_id": str(row["document_id"]),
                "category": str(row["category"]),
                "title": str(row["title"]),
                "content": str(row.get("content", "")),
                "keywords": str(row.get("keywords", "")),
                "similarity_score": round(float(scores[idx]), 6),
            })
        return results

    def search(self, query, model="FastText", top_k=DEFAULT_TOP_K):
        if not self.ready:
            raise RuntimeError(
                self.store.error or "Search engine is not ready."
            )
        if model not in self.available_models:
            raise ValueError(f"Model must be one of {self.available_models}")
        top_k = max(1, min(int(top_k), len(self.documents)))

        if model == "TF-IDF":
            qvec = self.store.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(qvec, self.tfidf_documents).flatten()
        else:
            qvec = self._semantic_query_vector(query, model)
            matrix = (
                self.word2vec_documents
                if model == "Word2Vec"
                else self.fasttext_documents
            )
            scores = self._safe_cosine(qvec, matrix)

        return {
            "query": query,
            "model": model,
            "top_k": top_k,
            "results": self._rank_results(scores, top_k),
        }

    def compare_models(self, query, top_k=5):
        return {
            "query": query,
            "top_k": top_k,
            "models": {
                model: self.search(query, model, top_k)["results"]
                for model in self.available_models
            },
        }
