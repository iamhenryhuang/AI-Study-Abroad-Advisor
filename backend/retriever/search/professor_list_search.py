"""Professor list retrieval: query professors (seed data written by db/loaders/load_professors.py).

Answers list-type questions such as "which professors does a school have"; it is complementary to professor_fetcher (a named professor's research details
through SerpAPI/Google Scholar): list the names first, then the user picks one for a deeper look.
"""
from __future__ import annotations

from retriever.search.db_query import fetch_dicts, readonly_connection


def _format_professor_text(row: dict) -> str:
    areas = row.get("research_areas") or []
    areas_text = "、".join(areas) if areas else "未提供"
    title = row.get("title")
    title_text = f"{title}｜" if title else ""
    return f"{row['name']}（{title_text}研究領域：{areas_text}）"


def professor_list_search(school_id: str, limit: int = 30) -> list[dict]:
    """Query a school's professor list and return chunk-format docs (type='professor_list')."""
    if not school_id:
        return []

    with readonly_connection() as conn:
        if not conn:
            print("[ProfessorListSearch] Could not get a database connection")
            return []

        try:
            rows = fetch_dicts(
                conn,
                """
                SELECT p.name, p.title, p.research_areas, p.profile_url, u.school_id
                FROM professors p
                JOIN universities u ON p.university_id = u.id
                WHERE u.school_id = %(school_id)s
                ORDER BY p.name
                LIMIT %(limit)s
                """,
                {"school_id": school_id, "limit": limit},
            )
        except Exception as e:
            print(f"[ProfessorListSearch] Query failed: {e}")
            return []

    docs = []
    for row in rows:
        docs.append({
            "type":        "professor_list",
            "chunk_text":  _format_professor_text(row),
            "source_url":  row.get("profile_url") or "",
            "school_id":   row.get("school_id") or "",
        })
    return docs
