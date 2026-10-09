# Architecture

Browser UI
   |
   | HTTP JSON
   v
FastAPI
   |
   +--> /api/health
   +--> /api/model-info
   +--> /api/predict
   +--> /api/explain
   |
   v
Text preprocessing
   |
   +--> title + body
   +--> Reuters shortcut removal
   +--> URL removal
   +--> whitespace normalization
   |
   v
Hugging Face tokenizer
   |
   v
DistilBERT sequence classifier
   |
   v
FAKE / REAL probabilities
   |
   v
Frontend result + confidence + influence signals

Model artifact:
models/fake_news_model/
