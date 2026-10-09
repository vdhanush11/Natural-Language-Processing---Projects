import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text

    t = re.sub(
        r'^[A-Z][A-Za-z\.,/\-\s]{0,40}\(Reuters\)\s*[-–—]\s*',
        '',
        t,
    )
    t = re.sub(r'\(Reuters\)', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bReuters\b', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'http\S+|www\.\S+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    return t


def build_content(title: str, text: str) -> str:
    title = title or ""
    text = text or ""
    combined = f"{title}. {text}".strip()
    return clean_text(combined)
