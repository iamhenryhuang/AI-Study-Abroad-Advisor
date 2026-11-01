"""
Full-text fallback: when text-to-SQL finds nothing in the programs family,
run a PostgreSQL full-text search over document_chunks.fts_vector instead (no embedding model needed).

If semantic vector retrieval is added later, it only needs one more cosine-similarity query on the same table;
the document_chunks.embedding column and the HNSW index are already in the schema.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from retriever.applicant_search import _EN_STOPWORDS
from retriever.db_query import fetch_dicts, readonly_connection

# The schema uses the 'simple' text search config (no stopword list and no stemming, so Chinese is not hurt by English rules),
# which means question words in the user's natural sentence (how/does/many...) are also treated as required keywords.
# The AND semantics of plainto_tsquery almost never match here, so OR semantics (any one word matching) are used as a wide fallback.
# English function words (what/are/the...) are filtered first, otherwise OR + ts_rank favors chunks with "more words" over "more relevant" ones.
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _build_or_tsquery(text: str) -> str | None:
    """Split a natural sentence into words and join them with OR into a tsquery string (parsed by to_tsquery('simple', ...))."""
    words = [w for w in _WORD_PATTERN.findall(text.lower())
             if len(w) > 1 and w not in _EN_STOPWORDS]
    if not words:
        return None
    return " | ".join(words)


def fulltext_search(query: str, school_id: str | None = None, limit: int = 5) -> list[dict]:
    """
    Run a full-text search over document_chunks and return the top `limit` by relevance.
    The shape follows the professor chunk data (chunk_text format) and can be fed straight to format_context_for_prompt.
    """
    tsquery_str = _build_or_tsquery(query)
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
            return [r for r in results if r.get("chunk_text")]
        except Exception as e:
            print(f"[FulltextSearch] Query failed: {e}")
            return []
