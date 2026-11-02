"""BGE-reranker-v2-m3 CrossEncoder reranking (the second stage of hybrid search).

CrossEncoder is imported lazily;
when the package or model is unavailable the caller (hybrid_search) catches the exception and falls back to the RRF order.
"""
from __future__ import annotations

import os
from pathlib import Path

from generator.client import sanitize_ssl_env

_model = None


def _resolve_model_id() -> str:
    """Use the local path when it exists (including resolving snapshots/ of the HF hub cache); otherwise download by hub id."""
    raw = os.getenv("BGE_RERANKER_MODEL_PATH", "")
    if not raw:
        return "BAAI/bge-reranker-v2-m3"
    model_path = Path(raw)
    if not model_path.exists():
        print(f"[reranker] Local model {model_path} not found; downloading from HuggingFace ...")
        return "BAAI/bge-reranker-v2-m3"
    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            resolved = snapshots[-1]
            print(f"[reranker] HF hub cache detected; using snapshot: {resolved}")
            return str(resolved)
    print(f"[reranker] Loading the local rerank model: {model_path}")
    return str(model_path)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        sanitize_ssl_env()
        _model = CrossEncoder(_resolve_model_id(), trust_remote_code=True)
    return _model


def rerank(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """Rerank documents that have chunk_text and return the top top_n (with rerank_score, descending)."""
    if not documents:
        return []
    model = _get_model()
    pairs = [(query, doc["chunk_text"]) for doc in documents]
    scores = model.predict(pairs)
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)
    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
    return ranked[:top_n]
