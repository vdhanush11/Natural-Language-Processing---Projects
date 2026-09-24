# ============================================================
# app/preprocessing.py
# ============================================================

import re

import nltk

from nltk.tokenize import word_tokenize


# ============================================================
# DOWNLOAD NLTK RESOURCES
# ============================================================

try:

    nltk.data.find(
        "tokenizers/punkt"
    )

except LookupError:

    nltk.download(
        "punkt",
        quiet=True
    )


try:

    nltk.data.find(
        "tokenizers/punkt_tab"
    )

except LookupError:

    nltk.download(
        "punkt_tab",
        quiet=True
    )


# ============================================================
# LEGAL TEXT TOKENIZATION
# ============================================================

def tokenize_legal_text(
    text: str
):

    # --------------------------------------------------------
    # Convert to string
    # --------------------------------------------------------

    text = str(text)


    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    text = text.lower()


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = word_tokenize(
        text
    )


    # --------------------------------------------------------
    # Keep alphabetic words
    # --------------------------------------------------------

    tokens = [

        token

        for token in tokens

        if token.isalpha()

    ]


    return tokens