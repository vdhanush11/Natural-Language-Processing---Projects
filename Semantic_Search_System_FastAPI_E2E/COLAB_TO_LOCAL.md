# Colab → FastAPI local mapping

Copy the generated files exactly as follows:

```text
Colab models/word2vec_embeddings.npy       → models/word2vec_embeddings.npy
Colab models/fasttext_word_embeddings.npy  → models/fasttext_word_embeddings.npy
Colab models/fasttext_ngram_embeddings.npy → models/fasttext_ngram_embeddings.npy
Colab models/word_to_idx.pkl                → models/word_to_idx.pkl
Colab models/idx_to_word.pkl                → models/idx_to_word.pkl
Colab models/ngram_to_idx.pkl               → models/ngram_to_idx.pkl
Colab models/tfidf_vectorizer.pkl            → models/tfidf_vectorizer.pkl
Colab models/test_word2vec_documents.npy    → models/test_word2vec_documents.npy
Colab models/test_fasttext_documents.npy     → models/test_fasttext_documents.npy
Colab outputs/test_processed.csv             → outputs/test_processed.csv
Colab results/model_comparison.csv           → results/model_comparison.csv
```

Then:

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_setup.py
python app.py
```

Open:

`http://127.0.0.1:8000`

FastAPI Swagger documentation:

`http://127.0.0.1:8000/docs`
