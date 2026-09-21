# Lifecycle stage 9 — Model Deployment (hand-off)
from __future__ import annotations

import os
from pathlib import Path

import joblib
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data.preprocess import clean_text, create_sentiment

app = FastAPI(title="Customer Feedback Analyzer")

base_dir = Path(__file__).resolve().parents[2]
model = joblib.load(base_dir / "models" / "sentiment_model.pkl")
vectorizer = joblib.load(base_dir / "models" / "tfidf_vectorizer.pkl")


class Review(BaseModel):
    text: str


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Customer Feedback Analyzer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .card {
            width: min(700px, 90vw);
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        h1 { margin-top: 0; }
        textarea {
            width: 100%;
            min-height: 150px;
            border-radius: 12px;
            border: 1px solid #475569;
            background: #0f172a;
            color: white;
            padding: 14px;
            resize: vertical;
            box-sizing: border-box;
            font-size: 16px;
        }
        button {
            margin-top: 12px;
            background: #38bdf8;
            color: #082f49;
            border: none;
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
            font-weight: 700;
        }
        .result {
            margin-top: 20px;
            padding: 16px;
            border-radius: 12px;
            background: #111827;
            border: 1px solid #334155;
        }
        .sentiment {
            display: inline-block;
            font-weight: 700;
            padding: 6px 12px;
            border-radius: 999px;
            text-transform: capitalize;
        }
        .positive { background: rgba(34,197,94,0.2); color: #86efac; }
        .negative { background: rgba(239,68,68,0.2); color: #fca5a5; }
        .neutral { background: rgba(250,204,21,0.2); color: #fcd34d; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Customer Feedback Analyzer</h1>
        <textarea id="reviewText" placeholder="Type customer feedback here..."></textarea>
        <button id="analyzeBtn">Analyze Sentiment</button>
        <div class="result">
            <strong>Sentiment:</strong>
            <span id="sentimentBadge" class="sentiment neutral">-</span>
            <div id="resultText" style="margin-top: 10px;">Waiting for feedback...</div>
        </div>
    </div>

    <script>
        const analyzeBtn = document.getElementById('analyzeBtn');
        const reviewText = document.getElementById('reviewText');
        const sentimentBadge = document.getElementById('sentimentBadge');
        const resultText = document.getElementById('resultText');

        analyzeBtn.addEventListener('click', async () => {
            const text = reviewText.value.trim();
            if (!text) {
                resultText.textContent = 'Please enter a review before analyzing.';
                return;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();
                const sentiment = data.sentiment || 'neutral';
                const confidence = (data.confidence || 0) * 100;
                sentimentBadge.textContent = sentiment;
                sentimentBadge.className = 'sentiment ' + sentiment;
                resultText.textContent = 'Customer sentiment: ' + sentiment + ' | Confidence: ' + confidence.toFixed(1) + '%';
            } catch (error) {
                resultText.textContent = 'Something went wrong while analyzing the review.';
                sentimentBadge.textContent = 'Error';
                sentimentBadge.className = 'sentiment neutral';
            }
        });
    </script>
</body>
</html>
"""


def send_telegram_notification(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def score_review(text: str):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    model_probs = model.predict_proba(X)[0]
    model_label = model.predict(X)[0]

    classes = model.classes_
    label_index = list(classes).index(model_label)
    confidence = float(model_probs[label_index])

    rule_label = create_sentiment(text)

    if rule_label == "negative" and model_label != "negative":
        return "negative", min(0.92, max(confidence, 0.65)), True
    if rule_label == "positive" and model_label != "positive":
        return "positive", min(0.92, max(confidence, 0.65)), True
    if rule_label == "neutral" and model_label in {"positive", "negative"} and confidence < 0.8:
        return "neutral", max(0.5, min(confidence, 0.8)), True

    return model_label, confidence, False


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


@app.post("/predict")
def predict(review: Review):
    text = review.text.strip()
    if not text:
        return {"sentiment": "neutral", "confidence": 0.0, "telegram_notification_sent": False}

    sentiment, confidence, _ = score_review(text)
    notification_sent = False

    if sentiment == "negative":
        notification_sent = send_telegram_notification(
            f"<b>Negative customer feedback received</b>\n\n{text}"
        )

    return {
        "sentiment": sentiment,
        "confidence": round(float(confidence), 4),
        "telegram_notification_sent": notification_sent,
    }


@app.get("/health")
def health():
    return {"status": "ok"}

