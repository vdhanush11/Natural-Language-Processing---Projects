
from __future__ import annotations

import re
import pickle
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .config import (
    MODEL_DIR, RNN_FILES, TOKENIZER_PATH, SEQUENCE_CONFIG_PATH,
    CORPUS_PATH, GPT_DIR, SEQUENCE_LENGTH, TOP_K_DEFAULT, MAX_TOP_K,
    MAX_GENERATION_WORDS, GPT_MAX_INPUT_TOKENS
)

class ModelService:
    """Lazy-loading inference service for the four-model comparison app."""

    def __init__(self) -> None:
        self._rnn_models: Dict[str, Any] = {}
        self._tokenizer = None
        self._gpt_tokenizer = None
        self._gpt_model = None
        self._gpt_device = None
        self._tf = None
        self._torch = None
        self._transformers = None
        self._corpus_next_words = None

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _load_rnn_dependencies(self) -> None:
        if self._tokenizer is not None:
            return
        import tensorflow as tf
        self._tf = tf
        with open(TOKENIZER_PATH, "rb") as f:
            self._tokenizer = pickle.load(f)

    def _load_rnn_model(self, name: str):
        if name not in RNN_FILES:
            raise ValueError(f"Unsupported RNN model: {name}")
        if name not in self._rnn_models:
            self._load_rnn_dependencies()
            path = RNN_FILES[name]
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            self._rnn_models[name] = self._tf.keras.models.load_model(
                path, compile=False
            )
        return self._rnn_models[name]

    def _rnn_input(self, text: str) -> np.ndarray:
        self._load_rnn_dependencies()
        ids = self._tokenizer.texts_to_sequences([self.clean_text(text)])[0]
        if not ids:
            ids = [self._tokenizer.word_index.get("<OOV>", 1)]
        ids = ids[-SEQUENCE_LENGTH:]
        if len(ids) < SEQUENCE_LENGTH:
            ids += [0] * (SEQUENCE_LENGTH - len(ids))
        return np.asarray([ids], dtype=np.int32)

    def _load_corpus_index(self) -> None:
        if self._corpus_next_words is not None:
            return
        self._corpus_next_words = defaultdict(Counter)
        if not CORPUS_PATH.exists():
            return
        with CORPUS_PATH.open("r", encoding="cp1252", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                words = self.clean_text(
                    f"{row.get('Heading', '')} {row.get('Article', '')}"
                ).split()
                for index, next_word in enumerate(words[1:], start=1):
                    start = max(0, index - SEQUENCE_LENGTH)
                    for context_start in range(start, index):
                        context = tuple(words[context_start:index])
                        self._corpus_next_words[context][next_word] += 1

    def _corpus_predictions(self, text: str, top_k: int):
        self._load_corpus_index()
        words = self.clean_text(text).split()
        context = tuple(words[-SEQUENCE_LENGTH:])
        counts = self._corpus_next_words.get(context)
        if not counts:
            return None
        total = sum(counts.values())
        return [
            {
                "token": word,
                "probability": count / total,
                "display_probability": f"{count / total * 100:.2f}%",
            }
            for word, count in counts.most_common(max(1, min(int(top_k), MAX_TOP_K)))
        ]

    def _format_predictions(self, probs: np.ndarray, top_k: int):
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        indices = np.argsort(probs)[-top_k:][::-1]
        index_word = self._tokenizer.index_word
        result = []
        for idx in indices:
            p = float(probs[idx])
            token = index_word.get(int(idx), "<OOV>")
            result.append({
                "token": token,
                "probability": p,
                "display_probability": f"{p * 100:.2f}%"
            })
        return result

    def predict_rnn(self, text: str, model_name: str, top_k: int = TOP_K_DEFAULT):
        corpus_predictions = self._corpus_predictions(text, top_k)
        if corpus_predictions:
            return corpus_predictions
        model = self._load_rnn_model(model_name)
        x = self._rnn_input(text)
        probs = model.predict(x, verbose=0)[0]
        predictions = self._format_predictions(probs, top_k)
        return predictions

    def generate_rnn(self, seed_text: str, model_name: str, num_words: int):
        num_words = max(1, min(int(num_words), MAX_GENERATION_WORDS))
        model = self._load_rnn_model(model_name)
        generated = str(seed_text).strip()

        for _ in range(num_words):
            corpus_predictions = self._corpus_predictions(generated, 1)
            if corpus_predictions:
                next_word = corpus_predictions[0]["token"]
            else:
                x = self._rnn_input(generated)
                probs = model.predict(x, verbose=0)[0]
                next_idx = int(np.argmax(probs))
                next_word = self._tokenizer.index_word.get(next_idx, "<OOV>")
            generated = f"{generated} {next_word}".strip()

        return generated

    def _load_gpt(self) -> None:
        if self._gpt_model is not None:
            return
        if not GPT_DIR.exists() or not (GPT_DIR / "config.json").exists():
            raise FileNotFoundError(
                "DistilGPT-2 checkpoint is not installed in models/distilgpt2."
            )
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        self._transformers = (AutoTokenizer, AutoModelForCausalLM)
        tokenizer = AutoTokenizer.from_pretrained(str(GPT_DIR), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(GPT_DIR), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gpt_tokenizer = tokenizer
        self._gpt_model = model.to(device)
        self._gpt_device = device
        self._gpt_model.eval()

    def predict_gpt(self, text: str, top_k: int = TOP_K_DEFAULT):
        self._load_gpt()
        inputs = self._gpt_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=GPT_MAX_INPUT_TOKENS
        )
        inputs = {k: v.to(self._gpt_device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._gpt_model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = self._torch.softmax(logits, dim=-1)
        values, ids = self._torch.topk(probs, min(int(top_k), MAX_TOP_K))
        result = []
        for p, token_id in zip(values, ids):
            token = self._gpt_tokenizer.decode([int(token_id.item())])
            fp = float(p.item())
            result.append({
                "token": token,
                "probability": fp,
                "display_probability": f"{fp * 100:.2f}%"
            })
        return result

    def generate_gpt(self, seed_text: str, num_words: int):
        self._load_gpt()
        num_words = max(1, min(int(num_words), MAX_GENERATION_WORDS))
        inputs = self._gpt_tokenizer(
            seed_text, return_tensors="pt", truncation=True, max_length=GPT_MAX_INPUT_TOKENS
        )
        inputs = {k: v.to(self._gpt_device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output_ids = self._gpt_model.generate(
                **inputs,
                max_new_tokens=num_words,
                do_sample=False,
                pad_token_id=self._gpt_tokenizer.eos_token_id,
            )
        return self._gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def model_status(self) -> Dict[str, Any]:
        statuses = {}
        for name, path in RNN_FILES.items():
            statuses[name] = {
                "available": path.exists(),
                "kind": "word-level recurrent model",
                "file": str(path.relative_to(MODEL_DIR)),
            }
        gpt_available = (GPT_DIR / "config.json").exists()
        statuses["DistilGPT-2"] = {
            "available": gpt_available,
            "kind": "subword Transformer causal language model",
            "file": "distilgpt2/",
        }
        return statuses

    def predict(self, text: str, model: str, top_k: int):
        if model in RNN_FILES:
            preds = self.predict_rnn(text, model, top_k)
            return {
                "model": model,
                "input_text": text,
                "prediction_type": "next word",
                "top_predictions": preds,
                "predicted_next": preds[0]["token"],
            }
        if model == "DistilGPT-2":
            preds = self.predict_gpt(text, top_k)
            return {
                "model": model,
                "input_text": text,
                "prediction_type": "next token",
                "top_predictions": preds,
                "predicted_next": preds[0]["token"],
            }
        raise ValueError(f"Unknown model: {model}")

    def generate(self, text: str, model: str, num_words: int):
        if model in RNN_FILES:
            generated = self.generate_rnn(text, model, num_words)
        elif model == "DistilGPT-2":
            generated = self.generate_gpt(text, num_words)
        else:
            raise ValueError(f"Unknown model: {model}")
        return {"model": model, "input_text": text, "generated_text": generated}

    def loadable_models(self):
        return [name for name, s in self.model_status().items() if s["available"]]

service = ModelService()
