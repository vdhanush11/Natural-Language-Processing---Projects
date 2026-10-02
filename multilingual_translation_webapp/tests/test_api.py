# Lightweight tests that do not require loading the 2.46 GB model.
# Run with: pytest tests/test_api.py

from backend.translator import LANGUAGES, SUPPORTED_PAIRS, protect_terms, restore_terms, postprocess

def test_language_registry():
    assert len(LANGUAGES) == 202
    assert LANGUAGES["eng_Latn"]["name"] == "English"
    assert LANGUAGES["tam_Taml"]["name"] == "Tamil"
    assert LANGUAGES["ace_Arab"]["name"] == "Achinese (Arabic script)"
    assert LANGUAGES["ajp_Arab"]["name"] == "South Levantine Arabic"
    assert all("_" not in info["name"] for info in LANGUAGES.values())
    assert ("eng_Latn", "tam_Taml") in SUPPORTED_PAIRS

def test_term_protection():
    text = "Payment failed: ERR-500. Contact test@example.com or https://example.com/a."
    masked, mapping = protect_terms(text)
    assert "ERR-500" not in masked
    assert "test@example.com" not in masked
    assert "https://example.com/a." not in masked
    restored = restore_terms(masked, mapping)
    assert "ERR-500" in restored
    assert "test@example.com" in restored

def test_postprocess():
    assert postprocess("  hello   world  ") == "hello world"
