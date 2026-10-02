# Multilingual Translation Web App

A production-oriented local web application for the **Multilingual Customer-Support Translation System** from the supplied technical assessment and notebook.

## What this project uses

The app uses **`facebook/nllb-200-distilled-600M`**, a pretrained multilingual Transformer encoder-decoder. The app exposes the complete NLLB language registry: 202 language/script entries supported by the tokenizer, with every language available as a source or target.

- English ↔ French
- English ↔ Spanish
- English ↔ Hindi
- English ↔ Tamil

The inference pipeline includes language detection, NLLB language-code mapping, technical-term protection, sub-word tokenisation, beam-search decoding, and post-processing.

> Important assessment accuracy note: the supplied notebook is an **inference-based pretrained NLLB pipeline**. It does not show task-specific fine-tuning of NLLB in the supplied implementation. The assessment asks candidates to use transfer learning through a pretrained checkpoint and to explain it. Do not claim task-specific fine-tuning unless you actually performed it.

## Features

- Beautiful responsive single-page UI
- Auto source-language detection
- Manual source-language override
- Full NLLB language registry (202 language/script entries)
- Swap source/target
- Technical-term protection for URLs, emails, error codes, SKUs, IDs, tags and handles
- Beam-search controls
- Output length and repetition controls
- Length-penalty control
- Copy translation
- Download translation as `.txt`
- Character/token counters
- Latency and language-detection confidence
- Pipeline visibility panel
- Batch translation endpoint
- Text/CSV upload endpoint
- FastAPI Swagger/OpenAPI at `/docs`
- GPU/CPU detection
- Local model-folder support
- Graceful model-load status

## 1. Recommended hardware

The supplied notebook was run successfully on a Tesla T4 GPU. NLLB-200 distilled 600M is large, so a GPU is strongly recommended for interactive use.

CPU mode is supported, but inference can be much slower.

## 2. Get the model

### Option A — download automatically

Leave `MODEL_DIR` empty. On first startup, Transformers downloads:

`facebook/nllb-200-distilled-600M`

### Option B — use your already-downloaded model

If you downloaded the model from Google Colab/Hugging Face, place the complete model directory anywhere on your PC and set:

Windows PowerShell:
```powershell
$env:MODEL_DIR="C:\path\to\nllb-200-distilled-600M"
```

Windows CMD:
```cmd
set MODEL_DIR=C:\path\to\nllb-200-distilled-600M
```

Git Bash:
```bash
export MODEL_DIR="/c/path/to/nllb-200-distilled-600M"
```

Linux/macOS:
```bash
export MODEL_DIR="/path/to/nllb-200-distilled-600M"
```

The folder should contain the normal Hugging Face files such as `config.json`, tokenizer files and the model weights.

## 3. Create a virtual environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

Git Bash:
```bash
python -m venv .venv
source .venv/Scripts/activate
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

If you already have a CUDA-enabled PyTorch installation, keep that installation rather than replacing it unnecessarily.

## 5. Start the application

From the project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

## 6. First test

Try:

**Input**
`My payment failed with error ERR-500, please help urgently!`

Target:
`Hindi`

The technical code `ERR-500` is protected before generation and restored afterwards.

Other good tests:

- `Where is my refund?` → French
- `The app keeps crashing on startup.` → Spanish
- `I need to reset my password.` → Tamil
- `Mera payment fail ho gaya, please help me urgently!` → Hindi
- `Bonjour, my tracking number ne fonctionne pas.` → Hindi/Tamil/French

## 7. Architecture

```text
Browser
   │
   ▼
FastAPI
   │
   ├── Input validation
   ├── Source language detection
   ├── ISO → NLLB language-code mapping
   ├── Technical-term protection
   ├── SentencePiece tokenisation
   ├── NLLB Transformer encoder-decoder
   │      ├── Self-attention
   │      ├── Multi-head attention
   │      └── Positional information
   ├── Beam-search decoding
   ├── Protected-term restoration
   └── Whitespace post-processing
   │
   ▼
Translated output + metadata
```

## 8. Why these UI controls exist

The UI exposes generation controls because they directly correspond to the notebook's generation configuration:

- **Beam width**: quality/speed trade-off.
- **Max output tokens**: output-length safety cap.
- **No-repeat n-gram**: reduces repetitive decoding.
- **Length penalty**: adjusts preference for shorter/longer sequences.
- **Protect technical terms**: keeps support-domain tokens such as `ERR-500`, order IDs and URLs intact.

The notebook's default values are preserved: 5 beams, 512 max new tokens, no-repeat 3-gram, length penalty 1.0.

## 9. Important scope

The assessment requires at least four language pairs and asks for preservation of intent, sentiment and technical accuracy. The supplied notebook implements the first requirement through the four English-centered pairs and protects technical tokens. It does not provide a dedicated sentiment classifier or a separate hallucination detector, so this web app does not falsely claim those as independently verified components.

The assessment also requires measurable quality using BLEU/chrF/COMET and a short explanation video. Those are evaluation/submission activities rather than requirements for the interactive inference screen.

## 10. Project structure

```text
multilingual_translation_webapp/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── translator.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── models/
│   └── README.md
├── scripts/
│   └── download_model.py
├── tests/
│   └── test_api.py
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```

## 11. API endpoints

- `GET /api/health`
- `GET /api/languages`
- `GET /api/model`
- `POST /api/translate`
- `POST /api/translate/batch`
- `POST /api/translate/file`

FastAPI automatically provides OpenAPI/Swagger documentation at `/docs`.

## 12. Security/production hardening

Before exposing this service to the public internet, add authentication, rate limiting, request-size limits at the reverse proxy, structured logging, HTTPS, CORS allowlisting, and secrets management.

For a local VS Code demonstration, the included setup is intentionally simple.

