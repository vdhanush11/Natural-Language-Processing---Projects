# Legal Clause Similarity Engine

## Project Objective

A semantic legal clause retrieval system that accepts a legal clause,
sentence, or paragraph and retrieves the most semantically similar
clauses from a predefined legal clause dataset.

## NLP Approach

The system uses:

1. Word2Vec
2. TF-IDF
3. TF-IDF Weighted Word2Vec
4. Cosine Similarity

## Dataset

Number of legal clauses:

21,070

## Model

Word2Vec vector size:

100

Vocabulary size:

8,864

## Retrieval Pipeline

User Query
↓
Tokenization
↓
TF-IDF
↓
Word2Vec
↓
TF-IDF Weighted Query Vector
↓
Cosine Similarity
↓
Top-K Similar Clauses

## Backend

FastAPI

## Frontend

HTML
CSS
JavaScript

## Run Application

Activate virtual environment:

```powershell
.venv\Scripts\activate