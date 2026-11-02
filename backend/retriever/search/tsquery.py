"""Build the OR tsquery strings used by the PostgreSQL full-text searches.

The schema uses the 'simple' text search config (no stopword list and no stemming, so Chinese is not hurt
by English rules), which means question words in a natural sentence (how/does/many...) would be treated as
required keywords. AND semantics almost never match here, so OR semantics (any one word matching) are used
as a wide fallback. English function words are filtered first, otherwise OR + ts_rank favors chunks with
"more words" over "more relevant" ones.

The tokens contain only letters and digits, so the result is safe to pass to to_tsquery('simple', ...).
"""
from __future__ import annotations

import re

EN_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "how", "when", "where", "why",
    "do", "does", "did", "done", "can", "could", "may", "might", "must",
    "shall", "should", "will", "would", "has", "have", "had",
    "for", "of", "in", "on", "at", "to", "from", "by", "with", "about",
    "as", "and", "or", "not", "no", "nor", "but", "if", "so", "than", "then",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "there", "here", "i", "you", "we", "he", "she", "my", "your", "our", "me", "us",
    "also", "just", "any", "some", "such", "too", "very", "please",
}

# Letters of any script. PostgreSQL's parser keeps a run of Chinese as one token, and so does this pattern,
# so a Chinese word only matches an identical run.
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def build_or_tsquery(text: str) -> str | None:
    """Split a natural sentence into words and join them with OR."""
    words = [w for w in _WORD_PATTERN.findall(text.lower())
             if len(w) > 1 and w not in EN_STOPWORDS]
    return " | ".join(words) or None
