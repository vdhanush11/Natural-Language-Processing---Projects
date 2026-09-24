# ============================================================
# tests/test_search.py
# ============================================================

from app.search_engine import (
    search_similar_clauses
)


def test_search_returns_results():

    query = (
        "The parties must protect "
        "confidential information."
    )


    results = search_similar_clauses(

        query=query,

        top_k=5

    )


    assert len(results) == 5


    assert (
        "clause"
        in results[0]
    )


    assert (
        "similarity"
        in results[0]
    )


    assert (
        "clause_type"
        in results[0]
    )


if __name__ == "__main__":

    test_search_returns_results()

    print(
        "✓ Search test passed!"
    )