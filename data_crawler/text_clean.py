"""Text cleaning.

- deduplicate_text: n-gram sliding-window deduplication, for the nav bar repeating across several DOM sources
- clean_noise: apply KEYWORDS_TO_REMOVE and remove_keywords from config/cleaning.py,
  cleaning before the text goes into chunks / the DB
"""
import hashlib

from .config import KEYWORDS_TO_REMOVE, remove_keywords


def normalize_spaces(text):
    return " ".join(text.split()) if text else ""


def deduplicate_text(text: str, window: int = 40) -> str:
    if not text:
        return text
    words = text.split()
    if len(words) < window * 2:
        return text

    seen: set = set()
    kept: list = []

    for i, word in enumerate(words):
        end = i + window
        if end <= len(words):
            ngram = tuple(w.lower() for w in words[i:end])
            if ngram in seen:
                continue
            seen.add(ngram)
        kept.append(word)

    return " ".join(kept)


def clean_noise(text: str) -> str:
    """Apply the keyword-removal list from config/cleaning.py."""
    if not text:
        return text
    return remove_keywords(text, KEYWORDS_TO_REMOVE)


def content_hash(text: str) -> str:
    """md5 content-hash (for cross-page deduplication)."""
    return hashlib.md5((text or "").encode("utf-8", errors="replace")).hexdigest()
