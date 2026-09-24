# ============================================================
# app/search_engine.py
# ============================================================

import numpy as np

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from app.model_loader import (
    word2vec_model,
    clause_vectors,
    tfidf_vectorizer,
    legal_docs
)

from app.preprocessing import (
    tokenize_legal_text
)


# ============================================================
# CREATE QUERY VECTOR
# ============================================================

def create_query_vector(
    query: str
):

    # --------------------------------------------------------
    # Tokenize query
    # --------------------------------------------------------

    tokens = tokenize_legal_text(
        query
    )


    # --------------------------------------------------------
    # Convert tokens into text
    # --------------------------------------------------------

    tokenized_query = " ".join(
        tokens
    )


    # --------------------------------------------------------
    # TF-IDF representation
    # --------------------------------------------------------

    tfidf_values = (
        tfidf_vectorizer
        .transform(
            [tokenized_query]
        )
        .toarray()[0]
    )


    # --------------------------------------------------------
    # TF-IDF vocabulary
    # --------------------------------------------------------

    feature_names = (
        tfidf_vectorizer
        .get_feature_names_out()
    )


    word_to_index = {

        word: index

        for index, word
        in enumerate(
            feature_names
        )

    }


    # --------------------------------------------------------
    # Collect Word2Vec vectors
    # --------------------------------------------------------

    vectors = []

    weights = []


    for word in tokens:

        # ----------------------------------------------------
        # Check whether word exists in both models
        # ----------------------------------------------------

        if (

            word in word2vec_model.wv

            and

            word in word_to_index

        ):

            index = (
                word_to_index[word]
            )

            weight = (
                tfidf_values[index]
            )

            vectors.append(
                word2vec_model.wv[word]
            )

            weights.append(
                weight
            )


    # --------------------------------------------------------
    # No recognized words
    # --------------------------------------------------------

    if len(vectors) == 0:

        return np.zeros(
            word2vec_model.vector_size,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    vectors = np.array(
        vectors
    )

    weights = np.array(
        weights
    )


    # --------------------------------------------------------
    # If all TF-IDF weights are zero
    # --------------------------------------------------------

    if weights.sum() == 0:

        return np.mean(
            vectors,
            axis=0
        ).astype(
            np.float32
        )


    # --------------------------------------------------------
    # TF-IDF weighted Word2Vec
    # --------------------------------------------------------

    query_vector = np.average(

        vectors,

        axis=0,

        weights=weights

    )


    return query_vector.astype(
        np.float32
    )


# ============================================================
# SEARCH SIMILAR CLAUSES
# ============================================================

def search_similar_clauses(
    query: str,
    top_k: int = 5,
    clause_type: str | None = None
):

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )


    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    top_k = int(top_k)

    if top_k < 1:

        top_k = 1

    if top_k > 20:

        top_k = 20


    # --------------------------------------------------------
    # Create query vector
    # --------------------------------------------------------

    query_vector = (
        create_query_vector(
            query
        )
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarity_scores = (
        cosine_similarity(
            [query_vector],
            clause_vectors
        )[0]
    )

    if clause_type and clause_type != "all":
        type_mask = legal_docs["clause_type"].str.casefold() == clause_type.casefold()
        similarity_scores = np.where(type_mask, similarity_scores, -1)


    # --------------------------------------------------------
    # Get top K indexes
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]


    # --------------------------------------------------------
    # Prepare results
    # --------------------------------------------------------

    results = []


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        score = float(
            similarity_scores[index]
        )


        result = {

            "rank":
                rank,

            "clause_id":
                str(
                    legal_docs.iloc[index][
                        "Unnamed: 0"
                    ]
                ),

            "clause_type":
                str(
                    legal_docs.iloc[index][
                        "clause_type"
                    ]
                ),

            "similarity":
                round(
                    score,
                    4
                ),

            "similarity_percentage":
                round(
                    score * 100,
                    2
                ),

            "clause":
                str(
                    legal_docs.iloc[index][
                        "clause_text"
                    ]
                )

        }


        results.append(
            result
        )


    return results