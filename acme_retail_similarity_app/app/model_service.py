from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import (
    DATA_PATH,
    FASTTEXT_EPOCHS,
    FASTTEXT_MIN_COUNT,
    FASTTEXT_VECTOR_SIZE,
    FASTTEXT_WINDOW,
    FASTTEXT_WORKERS,
    MODEL_DIR,
    TFIDF_MAX_FEATURES,
    W2V_EPOCHS,
    W2V_MIN_COUNT,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_WORKERS,
)
from .preprocessing import clean_text, prepare_catalogue


class RecommendationService:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.df: pd.DataFrame | None = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.word2vec = None
        self.fasttext = None
        self.word2vec_embeddings = None
        self.fasttext_embeddings = None
        self.status = {
            "catalogue": False,
            "TF-IDF": False,
            "Word2Vec": False,
            "FastText": False,
        }

    def initialize(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._load_catalogue()
        self._load_artifacts()
        if not self.status["TF-IDF"]:
            self.train("TF-IDF")

    def _load_catalogue(self) -> None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}. "
                "Set ACME_DATA_PATH or place the CSV in data/."
            )
        raw = pd.read_csv(DATA_PATH)
        self.df = prepare_catalogue(raw)
        self.status["catalogue"] = True

    def _load_artifacts(self) -> None:
        catalogue_path = MODEL_DIR / "catalogue.pkl"
        if catalogue_path.exists():
            try:
                cached = pd.read_pickle(catalogue_path)
                if len(cached) == len(self.df):
                    self.df = cached
            except Exception:
                pass

        tfidf_vec = MODEL_DIR / "tfidf_vectorizer.joblib"
        tfidf_mat = MODEL_DIR / "tfidf_matrix.joblib"
        if tfidf_vec.exists() and tfidf_mat.exists():
            try:
                self.tfidf_vectorizer = joblib.load(tfidf_vec)
                self.tfidf_matrix = joblib.load(tfidf_mat)
                if self.tfidf_matrix.shape[0] == len(self.df):
                    self.status["TF-IDF"] = True
            except Exception:
                self.status["TF-IDF"] = False

        w2v_path = MODEL_DIR / "word2vec.model"
        w2v_emb = MODEL_DIR / "word2vec_embeddings.npy"
        if w2v_path.exists() and w2v_emb.exists():
            try:
                self.word2vec = Word2Vec.load(w2v_path)
                self.word2vec_embeddings = np.load(w2v_emb)
                if self.word2vec_embeddings.shape[0] == len(self.df):
                    self.status["Word2Vec"] = True
            except Exception:
                self.status["Word2Vec"] = False

        ft_path = MODEL_DIR / "fasttext.model"
        ft_emb = MODEL_DIR / "fasttext_embeddings.npy"
        if ft_path.exists() and ft_emb.exists():
            try:
                self.fasttext = FastText.load(ft_path)
                self.fasttext_embeddings = np.load(ft_emb)
                if self.fasttext_embeddings.shape[0] == len(self.df):
                    self.status["FastText"] = True
            except Exception:
                self.status["FastText"] = False

    @staticmethod
    def document_vector(tokens, model) -> np.ndarray:
        vectors = []
        for token in tokens:
            try:
                vectors.append(model.wv[token])
            except KeyError:
                continue

        if not vectors:
            return np.zeros(model.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def train(self, model: str = "all") -> None:
        with self.lock:
            if self.df is None:
                self._load_catalogue()

            if model in ("TF-IDF", "all"):
                self._train_tfidf()

            if model in ("Word2Vec", "all"):
                self._train_word2vec()

            if model in ("FastText", "all"):
                self._train_fasttext()

            self.df.to_pickle(MODEL_DIR / "catalogue.pkl")

    def _train_tfidf(self) -> None:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=TFIDF_MAX_FEATURES,
        )
        matrix = vectorizer.fit_transform(self.df["model_text"])
        joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
        joblib.dump(matrix, MODEL_DIR / "tfidf_matrix.joblib")
        self.tfidf_vectorizer = vectorizer
        self.tfidf_matrix = matrix
        self.status["TF-IDF"] = True

    def _train_word2vec(self) -> None:
        sentences = self.df["tokens"].tolist()
        model = Word2Vec(
            vector_size=W2V_VECTOR_SIZE,
            window=W2V_WINDOW,
            min_count=W2V_MIN_COUNT,
            workers=W2V_WORKERS,
            sg=1,
            seed=42,
        )
        model.build_vocab(sentences)
        model.train(
            sentences,
            total_examples=len(sentences),
            epochs=W2V_EPOCHS,
        )

        embeddings = np.vstack([
            self.document_vector(tokens, model)
            for tokens in sentences
        ])
        embeddings = normalize(embeddings)

        model.save(MODEL_DIR / "word2vec.model")
        np.save(MODEL_DIR / "word2vec_embeddings.npy", embeddings)

        self.word2vec = model
        self.word2vec_embeddings = embeddings
        self.status["Word2Vec"] = True

    def _train_fasttext(self) -> None:
        sentences = self.df["tokens"].tolist()
        model = FastText(
            vector_size=FASTTEXT_VECTOR_SIZE,
            window=FASTTEXT_WINDOW,
            min_count=FASTTEXT_MIN_COUNT,
            workers=FASTTEXT_WORKERS,
            sg=1,
            min_n=3,
            max_n=6,
            seed=42,
        )
        model.build_vocab(sentences)
        model.train(
            sentences,
            total_examples=len(sentences),
            epochs=FASTTEXT_EPOCHS,
        )

        embeddings = np.vstack([
            self.document_vector(tokens, model)
            for tokens in sentences
        ])
        embeddings = normalize(embeddings)

        model.save(MODEL_DIR / "fasttext.model")
        np.save(MODEL_DIR / "fasttext_embeddings.npy", embeddings)

        self.fasttext = model
        self.fasttext_embeddings = embeddings
        self.status["FastText"] = True

    def ensure_model(self, model: str) -> None:
        if not self.status.get(model, False):
            self.train(model)

    def _query_from_input(
        self,
        query_text: str | None,
        catalog_id: int | None,
    ) -> tuple[str, int | None, dict[str, Any] | None]:
        if catalog_id is not None:
            matches = self.df[self.df["catalog_id"] == int(catalog_id)]
            if matches.empty:
                raise ValueError(f"Catalog ID {catalog_id} was not found.")
            row = matches.iloc[0]
            return row["model_text"], int(catalog_id), self._row_to_dict(row)

        if query_text and query_text.strip():
            return query_text.strip(), None, None

        raise ValueError("Provide either query_text or catalog_id.")

    def _row_to_dict(self, row) -> dict[str, Any]:
        def safe(value):
            if pd.isna(value):
                return None
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            return value

        return {
            "catalog_id": int(row["catalog_id"]),
            "name": safe(row.get("name")),
            "main_category": safe(row.get("main_category")),
            "sub_category": safe(row.get("sub_category")),
            "ratings": safe(row.get("ratings")),
            "no_of_ratings": safe(row.get("no_of_ratings")),
            "discount_price": safe(row.get("discount_price")),
            "actual_price": safe(row.get("actual_price")),
            "image": safe(row.get("image")),
            "link": safe(row.get("link")),
        }

    def _result_records(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        records = []
        for idx, score in zip(indices, scores):
            row = self.df.iloc[int(idx)]
            record = self._row_to_dict(row)
            record["similarity_score"] = round(float(score), 6)
            records.append(record)
        return records

    def recommend(
        self,
        query_text: str | None,
        catalog_id: int | None,
        model: str,
        top_n: int,
        main_category: str | None = None,
        sub_category: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_model(model)
        source_text, source_id, source_product = self._query_from_input(
            query_text, catalog_id
        )

        with self.lock:
            if model == "TF-IDF":
                q = self.tfidf_vectorizer.transform([clean_text(source_text)])
                scores = self.tfidf_matrix.dot(q.T).toarray().ravel()

            elif model == "Word2Vec":
                q = normalize(
                    self.document_vector(
                        clean_text(source_text).split(),
                        self.word2vec,
                    ).reshape(1, -1)
                )
                scores = (self.word2vec_embeddings @ q.T).ravel()

            elif model == "FastText":
                q = normalize(
                    self.document_vector(
                        clean_text(source_text).split(),
                        self.fasttext,
                    ).reshape(1, -1)
                )
                scores = (self.fasttext_embeddings @ q.T).ravel()

            else:
                raise ValueError(f"Unsupported model: {model}")

            mask = np.ones(len(self.df), dtype=bool)

            if source_id is not None:
                mask &= self.df["catalog_id"].to_numpy() != source_id

            if main_category:
                mask &= (
                    self.df["main_category"]
                    .fillna("")
                    .astype(str)
                    .eq(main_category)
                    .to_numpy()
                )

            if sub_category:
                mask &= (
                    self.df["sub_category"]
                    .fillna("")
                    .astype(str)
                    .eq(sub_category)
                    .to_numpy()
                )

            candidate_indices = np.flatnonzero(mask)
            candidate_scores = scores[candidate_indices]

            if candidate_indices.size == 0:
                return {
                    "model": model,
                    "query": source_product or {"text": source_text},
                    "recommendations": [],
                }

            k = min(int(top_n), candidate_indices.size)

            # Efficient partial sort for large catalogues.
            if k < candidate_indices.size:
                partial = np.argpartition(
                    -candidate_scores, k - 1
                )[:k]
                order = partial[
                    np.argsort(-candidate_scores[partial])
                ]
            else:
                order = np.argsort(-candidate_scores)

            selected_indices = candidate_indices[order]
            selected_scores = scores[selected_indices]

            return {
                "model": model,
                "query": source_product or {
                    "text": source_text,
                    "catalog_id": None,
                },
                "recommendations": self._result_records(
                    selected_indices,
                    selected_scores,
                ),
            }

    def compare(
        self,
        query_text: str | None,
        catalog_id: int | None,
        top_n: int,
    ) -> dict[str, Any]:
        results = {}
        for model in ["TF-IDF", "Word2Vec", "FastText"]:
            results[model] = self.recommend(
                query_text=query_text,
                catalog_id=catalog_id,
                model=model,
                top_n=top_n,
            )
        return results

    def products(
        self,
        search: str | None = None,
        main_category: str | None = None,
        sub_category: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        data = self.df

        if search:
            needle = clean_text(search)
            data = data[
                data["model_text"].str.contains(
                    needle, regex=False, na=False
                )
            ]

        if main_category:
            data = data[data["main_category"].eq(main_category)]

        if sub_category:
            data = data[data["sub_category"].eq(sub_category)]

        data = data.head(max(1, min(limit, 100)))
        return [self._row_to_dict(row) for _, row in data.iterrows()]

    def categories(self) -> dict[str, list[str]]:
        main = sorted(
            self.df["main_category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        sub = sorted(
            self.df["sub_category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        sub_categories_by_main = {}
        for category, group in self.df.groupby("main_category"):
            sub_categories_by_main[str(category)] = sorted(
                group["sub_category"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        return {
            "main_categories": main,
            "sub_categories": sub,
            "sub_categories_by_main": sub_categories_by_main,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "products": int(len(self.df)),
            "main_categories": int(
                self.df["main_category"].nunique(dropna=True)
            ),
            "sub_categories": int(
                self.df["sub_category"].nunique(dropna=True)
            ),
            "models": self.status.copy(),
        }


service = RecommendationService()
