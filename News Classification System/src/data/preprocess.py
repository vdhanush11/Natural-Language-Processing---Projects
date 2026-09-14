# Lifecycle stage 4 — Data Preparation (text cleaning)
 
import re
import spacy
 
nlp = spacy.load("en_core_web_sm")
 
 
def clean_text(text):
 
    text = str(text).lower()
 
    text = re.sub(r"http\S+", "", text)
 
    text = re.sub(r"[^a-zA-Z ]", "", text)
 
    doc = nlp(text)
 
    words = []
 
    for token in doc:
 
        if not token.is_stop:
            words.append(token.lemma_)
 
    return " ".join(words)
 
 
def clean_batch(texts):
 
    cleaned = []
 
    # nlp.pipe streams the texts through spaCy in batches;
    # disabling parser + ner skips the slow components we don't use
 
    docs = nlp.pipe(texts,
                    disable=["parser", "ner"],
                    batch_size=200)
 
    for doc in docs:
 
        words = [t.lemma_ for t in doc if not t.is_stop]
 
        cleaned.append(" ".join(words))
 
    return cleaned
