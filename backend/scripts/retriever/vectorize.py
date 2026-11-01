"""Query-side embedding: BGE-M3 (1024 dimensions), which hybrid_search uses to turn the user's question into a vector.

It shares the BGE_EMBED_MODEL_PATH env var with data_crawler/backfill_embeddings.py,
so the model only needs to be downloaded and stored once. sentence_transformers is imported lazily:
when the package or model is unavailable, the caller (the hybrid_search facade) degrades to plain FTS in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env explicitly: a bare load_dotenv() only looks for a .env in the CWD (the project root has none),
# and it used to work only as a side effect of other modules being imported first.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_BACKEND_ROOT / ".env")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_path = os.getenv("BGE_EMBED_MODEL_PATH") or "BAAI/bge-m3"
        print(f"[vectorize] Loading the embedding model: {model_path}")
        _model = SentenceTransformer(model_path, trust_remote_code=True)
    return _model


def embed_query(text: str) -> list[float]:
    """Turn one query string into a 1024-dimension vector; an empty string returns []."""
    if not text or not text.strip():
        return []
    embeddings = _get_model().encode(
        [text],
        batch_size=8,
        normalize_embeddings=True,   # Normalize the vectors to unit length (1), so the cosine distance <=> computed in SQL later is on the same scale as the vectors written by the backfill
        show_progress_bar=False,
    )
    return embeddings[0].tolist()
