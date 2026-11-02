"""Hybrid retrieval with vectors + FTS fused by RRF (an upgrade of the agent's text-retrieval fallback).

Three design points that match the current retrieval modules:
  1. The vector_matches CTE adds WHERE embedding IS NOT NULL
     (the pipeline defaults to ENABLE_EMBEDDING=off, so NULL vectors exist and would pollute the ranking unfiltered)
  2. conn.read_only = True (consistent with the current retrieval modules)
  3. The doc shape matches fulltext_search (chunk_text/source_url/school_id/university_name)

Error semantics: query/connection/embedding failures always raise (hybrid_search_with_fallback
degrades to plain FTS); [] is returned only when there is truly no data. Exception: if the reranker fails, fall back to the RRF order.
"""
from __future__ import annotations

import os

from database.connection import get_connection
from retriever.search.db_query import fetch_dicts, readonly_connection
from retriever.search.fulltext_search import fulltext_search
from retriever.search.reranker import rerank
from retriever.search.vectorize import embed_query

# RRF constant: the usual k=60 from the literature
_RRF_K = 60


def _to_pgvector_literal(vec: list[float]) -> str:
    """Convert to a pgvector text literal '[0.1,0.2,...]', avoiding psycopg type-adaptation problems."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _build_sql(school_id: str | None, initial_k: int) -> str:
    school_filter = "AND school_id = %(school_id)s" if school_id else ""
    return f"""
        WITH vector_matches AS (
            SELECT id,
                   1 - (embedding <=> %(vec)s::vector) AS vector_score,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank
            FROM document_chunks
            WHERE embedding IS NOT NULL
              {school_filter}
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT {initial_k}
        ),
        keyword_matches AS (
            SELECT id,
                   ts_rank_cd(fts_vector, websearch_to_tsquery('simple', %(q)s)) AS fts_score,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(fts_vector, websearch_to_tsquery('simple', %(q)s)) DESC
                   ) AS rank
            FROM document_chunks
            WHERE fts_vector @@ websearch_to_tsquery('simple', %(q)s)
              {school_filter}
            ORDER BY rank
            LIMIT {initial_k}
        )
        SELECT dc.chunk_text,
               dc.source_url,
               dc.school_id,
               u.name AS university_name,
               COALESCE(vm.vector_score, 0) AS vector_score,
               COALESCE(km.fts_score, 0)    AS fts_score,
               (COALESCE(1.0 / ({_RRF_K} + vm.rank), 0)
                + COALESCE(1.0 / ({_RRF_K} + km.rank), 0)) AS rrf_score
        FROM document_chunks dc
        LEFT JOIN universities u ON u.school_id = dc.school_id
        LEFT JOIN vector_matches vm ON dc.id = vm.id
        LEFT JOIN keyword_matches km ON dc.id = km.id
        WHERE vm.id IS NOT NULL OR km.id IS NOT NULL
        ORDER BY rrf_score DESC
        LIMIT {initial_k}
    """


def hybrid_search(
    query: str,
    school_id: str | None = None,
    limit: int = 5,
    use_rerank: bool = True,
) -> list[dict]:
    """RRF hybrid retrieval -> CrossEncoder rerank, returning the top `limit` docs.

    If the reranker fails, the RRF order is returned.
    """
    vec = embed_query(query)
    if not vec:
        return []

    conn = get_connection()
    if not conn:
        raise ConnectionError("[HybridSearch] Could not get a database connection")

    conn.read_only = True
    initial_k = limit * 2

    try:
        with conn.cursor() as cur:
            params = {"vec": _to_pgvector_literal(vec), "q": query}
            if school_id:
                params["school_id"] = school_id
            cur.execute(_build_sql(school_id, initial_k), params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    candidates = [{**dict(zip(columns, row)), "type": "official_page"} for row in rows]
    candidates = [c for c in candidates if c.get("chunk_text")]
    if not candidates:
        return []

    if use_rerank and len(candidates) > limit:
        try:
            return rerank(query, candidates, top_n=limit)
        except Exception as e:
            print(f"[HybridSearch] Reranking failed, using the RRF order: {e}")

    return candidates[:limit]


# --- Degrading facade (the agent's only entry point) ------------------------------

def _hybrid_search_enabled() -> bool:
    """The ENABLE_HYBRID_SEARCH switch (on by default). When set to false/0/off, skip vector retrieval entirely
    and do not try to load the embedding/reranker models -- in a development environment without the models,
    this avoids wasting time loading models on every query only to fail and degrade.
    """
    return os.getenv("ENABLE_HYBRID_SEARCH", "true").strip().lower() not in ("0", "false", "off")


def _has_embeddings(school_id: str | None) -> bool:
    """Whether any vector has been backfilled for this search. Embeddings are off by default in the crawler."""
    sql = "SELECT EXISTS (SELECT 1 FROM document_chunks WHERE embedding IS NOT NULL"
    params: dict = {}
    if school_id:
        sql += " AND school_id = %(school_id)s"
        params["school_id"] = school_id
    sql += ") AS has_embeddings"
    with readonly_connection() as conn:
        if not conn:
            return False
        try:
            return bool(fetch_dicts(conn, sql, params)[0]["has_embeddings"])
        except Exception as e:
            print(f"[HybridSearch] Could not check for embeddings: {e}")
            return False


def hybrid_search_with_fallback(
    query: str,
    school_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """The agent fallback's only entry point: try hybrid first, degrade to plain FTS otherwise, and never raise.

    Plain FTS is used when ENABLE_HYBRID_SEARCH=false, when no vectors have been backfilled (loading the model
    would download gigabytes for nothing), when hybrid fails (model not downloaded, sentence_transformers missing,
    DB error), and when hybrid finds nothing. The last case matters because the keyword leg of hybrid requires
    all the query words, while plain FTS accepts any of them.
    """
    if not _hybrid_search_enabled() or not _has_embeddings(school_id):
        return fulltext_search(query, school_id=school_id, limit=limit)

    try:
        return hybrid_search(query, school_id=school_id, limit=limit) \
            or fulltext_search(query, school_id=school_id, limit=limit)
    except Exception as e:
        print(f"[HybridSearch] Hybrid retrieval unavailable ({type(e).__name__}: {e}); degrading to plain FTS")
        return fulltext_search(query, school_id=school_id, limit=limit)
