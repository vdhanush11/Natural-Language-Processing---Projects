import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
QUARANTINE_FILE = QUARANTINE_DIR / "emails.jsonl"
LEGACY_QUARANTINE_FILE = QUARANTINE_DIR / "quarantine.csv"


def quarantine_email(text: str, label: str, confidence: float | None = None) -> dict[str, Any]:
    """Append a classified spam message to the local quarantine ledger."""
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": f"q_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "text": text,
        "label": label,
        "confidence": confidence,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
    }
    with QUARANTINE_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return record


def list_quarantined(limit: int = 50) -> list[dict[str, Any]]:
    """Return newest quarantine records, including records from the old CSV format."""
    records: list[dict[str, Any]] = []
    if QUARANTINE_FILE.exists():
        with QUARANTINE_FILE.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
    if LEGACY_QUARANTINE_FILE.exists():
        with LEGACY_QUARANTINE_FILE.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                records.append({
                    "id": f"legacy_{row.get('timestamp', '')}",
                    "text": row.get("email_text", ""),
                    "label": row.get("label", "spam"),
                    "confidence": None,
                    "quarantined_at": row.get("timestamp", ""),
                })
    return sorted(records, key=lambda item: item.get("quarantined_at", ""), reverse=True)[:limit]
