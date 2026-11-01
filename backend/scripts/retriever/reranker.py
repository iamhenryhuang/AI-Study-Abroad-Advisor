"""BGE-reranker-v2-m3 CrossEncoder reranking (the second stage of hybrid search).

CrossEncoder is imported lazily;
when the package or model is unavailable the caller (hybrid_search) catches the exception and falls back to the RRF order.
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


def _sanitize_ssl_env() -> None:
    """An SSL_CERT_FILE that points to a missing file makes the model download crash, so clear it first."""
    cert_file = os.getenv("SSL_CERT_FILE")
    if cert_file and not Path(cert_file).exists():
        print(f"[reranker] SSL_CERT_FILE is invalid and was ignored: {cert_file}")
        os.environ.pop("SSL_CERT_FILE", None)


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
        _sanitize_ssl_env()
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
