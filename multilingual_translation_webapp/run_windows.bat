@echo off
setlocal
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -r requirements.txt
if not defined MODEL_DIR set MODEL_DIR=
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
