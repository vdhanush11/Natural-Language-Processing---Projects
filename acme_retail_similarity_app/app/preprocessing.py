import html
import re
import unicodedata

import pandas as pd


def clean_text(text) -> str:
    """Same core text cleaning used in the supplied notebook."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    required = ["name", "main_category", "sub_category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy(deep=True)

    for column in required:
        out[f"{column}_clean"] = (
            out[column]
            .fillna("")
            .astype(str)
            .map(clean_text)
        )

    out["model_text"] = (
        out["name_clean"] + " "
        + out["name_clean"] + " "
        + out["sub_category_clean"] + " "
        + out["main_category_clean"]
    ).str.strip()

    out = out[out["model_text"].ne("")].copy()
    out.reset_index(drop=True, inplace=True)
    out["catalog_id"] = out.index.astype(int)
    out["tokens"] = out["model_text"].str.split()

    return out
