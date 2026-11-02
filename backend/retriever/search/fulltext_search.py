"""
Full-text fallback: when text-to-SQL finds nothing in the programs family,
run a PostgreSQL full-text search over document_chunks.fts_vector instead (no embedding model needed).

Vector retrieval on the same table lives in hybrid_search.py, which falls back to this module.
"""

from __future__ import annotations

from retriever.search.db_query import fetch_dicts, readonly_connection
from retriever.search.tsquery import build_or_tsquery


def fulltext_search(query: str, school_id: str | None = None, limit: int = 5) -> list[dict]:
    """
    Run a full-text search over document_chunks and return the top `limit` by relevance.
    The docs carry type='official_page' and can be fed straight to format_context_for_prompt.
    """
    tsquery_str = build_or_tsquery(query)
    if not tsquery_str:
        return []

    sql = """
        SELECT dc.school_id,
               u.name AS university_name,
               dc.chunk_text,
               dc.source_url,
               ts_rank(dc.fts_vector, to_tsquery('simple', %(q)s)) AS rank
        FROM document_chunks dc
        LEFT JOIN universities u ON u.school_id = dc.school_id
        WHERE dc.fts_vector @@ to_tsquery('simple', %(q)s)
    """
    params: dict = {"q": tsquery_str}
    if school_id:
        sql += " AND dc.school_id = %(school_id)s"
        params["school_id"] = school_id
    sql += " ORDER BY rank DESC LIMIT %(limit)s"
    params["limit"] = limit

    with readonly_connection() as conn:
        if not conn:
            print("[FulltextSearch] Could not get a database connection")
            return []

        try:
            results = fetch_dicts(conn, sql, params)
            return [{**r, "type": "official_page"} for r in results if r.get("chunk_text")]
        except Exception as e:
            print(f"[FulltextSearch] Query failed: {e}")
            return []
