# src/preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download once
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("punkt")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Basic text cleaning
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str):
    """
    Tokenize words
    """
    text = clean_text(text)
    return word_tokenize(text)


def remove_stopwords(tokens):
    """
    Remove common words
    """
    return [word for word in tokens if word not in stop_words]


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline
    """
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)