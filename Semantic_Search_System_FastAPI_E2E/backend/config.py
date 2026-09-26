from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULT_DIR = BASE_DIR / "results"

PROCESSED_TEST_FILE = OUTPUT_DIR / "test_processed.csv"

WORD2VEC_EMBEDDINGS = MODEL_DIR / "word2vec_embeddings.npy"
FASTTEXT_WORD_EMBEDDINGS = MODEL_DIR / "fasttext_word_embeddings.npy"
FASTTEXT_NGRAM_EMBEDDINGS = MODEL_DIR / "fasttext_ngram_embeddings.npy"

WORD_TO_IDX = MODEL_DIR / "word_to_idx.pkl"
IDX_TO_WORD = MODEL_DIR / "idx_to_word.pkl"
NGRAM_TO_IDX = MODEL_DIR / "ngram_to_idx.pkl"
TFIDF_VECTORIZER = MODEL_DIR / "tfidf_vectorizer.pkl"

TEST_WORD2VEC_DOCUMENTS = MODEL_DIR / "test_word2vec_documents.npy"
TEST_FASTTEXT_DOCUMENTS = MODEL_DIR / "test_fasttext_documents.npy"
MODEL_COMPARISON = RESULT_DIR / "model_comparison.csv"

MIN_NGRAM = 3
MAX_NGRAM = 6

AVAILABLE_MODELS = ["Word2Vec", "FastText", "TF-IDF"]
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
