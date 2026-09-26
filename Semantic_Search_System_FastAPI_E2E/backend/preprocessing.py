import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))

def preprocess_text(text):
    if text is None:
        return []
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [w for w in word_tokenize(text) if w not in STOP_WORDS]
