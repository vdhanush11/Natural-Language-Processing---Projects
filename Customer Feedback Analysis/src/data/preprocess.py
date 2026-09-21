# Lifecycle stage 4 — Data Preparation (sentiment labels)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import spacy
 
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
 
analyzer = SentimentIntensityAnalyzer()
 
 
def create_sentiment(text):
 
    score = analyzer.polarity_scores(str(text))["compound"]
 
    if score >= 0.05:
        return "positive"
 
    elif score <= -0.05:
        return "negative"
 
    else:
        return "neutral"


# Lifecycle stage 4 — Data Preparation (text cleaning)

 
nlp = spacy.load("en_core_web_sm")
 
# Negation words carry sentiment, so we never drop them
KEEP = {"not", "no", "never", "nor", "none", "nt"}
 
 
def clean_text(text):
 
    text = str(text).lower()
 
    text = re.sub(r"http\S+", "", text)
 
    text = re.sub(r"[^a-zA-Z ]", "", text)
 
    doc = nlp(text)
 
    words = []
 
    for token in doc:
 
        if token.lemma_ in KEEP or not token.is_stop:
            words.append(token.lemma_)
 
    return " ".join(words)
 
 
def clean_batch(texts):
 
    cleaned = []
 
    # nlp.pipe streams the texts through spaCy in batches;
    # disabling parser + ner skips the slow components we don’t use
    docs = nlp.pipe(
        texts,
        disable=["parser", "ner"],
        batch_size=200
    )
 
    for doc in docs:
 
        words = [
            t.lemma_ for t in doc
            if t.lemma_ in KEEP or not t.is_stop
        ]
 
        cleaned.append(" ".join(words))
 
    return cleaned
