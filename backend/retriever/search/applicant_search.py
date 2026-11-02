"""Application experience retrieval: combines applicant_reports and user_experiences.

The former is GradCafe / 1point3acres, the latter is the experiences shared by users on this site. Both are "experience" data --
unofficial and biased by who reports, which is a very different nature from programs (the official application thresholds).
Every returned doc is tagged type='applicant_experience' so the generator adds the unofficial warning and the guardrails.

It follows the document_chunks full-text approach: the 'simple' config plus an OR tsquery cast wide.
PostgreSQL's parser keeps a run of Chinese as one token, so Chinese words only match an identical run. A purely
Chinese question therefore usually finds no keyword hit and relies on the fallback to the school's latest reports.
"""
from __future__ import annotations

import re

from retriever.school_aliases import SCHOOL_ALIASES, detect_school_ids
from retriever.search.db_query import fetch_dicts, readonly_connection
from retriever.search.tsquery import build_or_tsquery

_USER_SOURCE_TERMS = (
    "使用者分享", "本站分享", "前端分享", "user submission", "user-submitted",
)
_EXTERNAL_SOURCE_TERMS = ("gradcafe", "一畝三分地", "一亩三分地", "1point3")


def user_submissions_only(query: str) -> bool:
    low = (query or "").lower()
    return any(term in low for term in _USER_SOURCE_TERMS) \
        and not any(term in low for term in _EXTERNAL_SOURCE_TERMS)


def external_reports_only(query: str) -> bool:
    low = (query or "").lower()
    return any(term in low for term in _EXTERNAL_SOURCE_TERMS) \
        and not any(term in low for term in _USER_SOURCE_TERMS)


def _format_report_text(row: dict) -> str:
    """Turn one applicant_reports row into a paragraph the LLM reads easily."""
    parts = []
    school = row.get("school_raw") or (row.get("school_id") or "").upper()
    if school:
        parts.append(f"學校：{school}")
    if row.get("program"):
        parts.append(f"科系：{row['program']}")
    if row.get("degree_level"):
        parts.append(f"學位：{row['degree_level']}")
    if row.get("decision"):
        parts.append(f"結果：{row['decision']}")
    if row.get("gpa") is not None:
        parts.append(f"GPA：{row['gpa']}")
    elif row.get("gpa_raw"):
        parts.append(f"GPA(原始)：{row['gpa_raw']}")
    if row.get("season"):
        parts.append(f"季度：{row['season']}")
    head = "；".join(parts)
    notes = (row.get("notes") or "").strip()
    return f"{head}\n{notes}".strip() if notes else head


def _format_user_experience_text(row: dict) -> str:
    """Turn an experience shared by a user of this site into the same searchable document format as external reports."""
    lines = ["來源：本站使用者分享（非官方）"]
    fields = (
        ("申請學校", row.get("apply_school")),
        ("申請科系", row.get("apply_program")),
        ("原就讀學校", row.get("graduate_school")),
        ("國家", row.get("country")),
        ("GPA", row.get("gpa")),
    )
    for label, value in fields:
        if value not in (None, ""):
            lines.append(f"{label}：{value}")

    rank = row.get("class_rank")
    size = row.get("class_size")
    if rank is not None and size is not None:
        lines.append(f"系排名：{rank}/{size}")

    items = []
    for item in row.get("experience") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("item") or "").strip()
        result = str(item.get("result") or "").strip()
        if name and result:
            items.append(f"{name}：{result}")
        elif name or result:
            items.append(name or result)
    if items:
        lines.append("經歷：" + "；".join(items))

    review = str(row.get("review") or "").strip()
    if review:
        lines.append("心得：" + review)
    return "\n".join(lines)


def _canonical_user_school_id(apply_school: str) -> str:
    detected = detect_school_ids(apply_school)
    return detected[0] if detected else ""


def _school_regexes(school_id: str) -> list[str]:
    """PostgreSQL regexes for a free-text school name, with the same matching rules as detect_school_ids:
    whole words for Latin aliases (\\y is a word boundary), substrings for Chinese ones."""
    return [rf"\y{re.escape(alias)}\y" if alias.isascii() else re.escape(alias)
            for alias in SCHOOL_ALIASES.get(school_id, [school_id.lower()])]


def _fetch_user_experiences(conn, school_id: str | None, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    sql = """
        SELECT id, graduate_school, country, apply_school, apply_program, gpa,
               class_rank, class_size, experience, review, created_at
        FROM user_experiences
    """
    params: dict = {"limit": limit}
    if school_id:
        sql += " WHERE lower(apply_school) ~ ANY(%(school_regexes)s)"
        params["school_regexes"] = _school_regexes(school_id)
    sql += " ORDER BY created_at DESC, id DESC LIMIT %(limit)s"
    return fetch_dicts(conn, sql, params)


def applicant_search(query: str, school_id: str | None = None, limit: int = 16) -> list[dict]:
    """Combine the experiences shared on this site with the external application reports, and return at most `limit` items.

    The shape follows the other chunk docs (chunk_text / source_url / school_id),
    and it carries type='applicant_experience' so the generator can recognize it and add the unofficial warning.
    """
    tsquery_str = build_or_tsquery(query)
    # Only when there are no usable keywords (for instance a generic Chinese question that yields no tokens) and no school is pinned is there truly nothing to look up
    if not tsquery_str and not school_id:
        return []

    _COLS = ("source, source_url, school_id, school_raw, program, "
             "degree_level, decision, gpa, gpa_raw, season, notes")

    with readonly_connection() as conn:
        if not conn:
            print("[ApplicantSearch] Could not get a database connection")
            return []

        user_rows: list[dict] = []
        if not external_reports_only(query):
            try:
                # Experiences shared on this site are kept first, so a large number of external cases does not push them out of the top `limit`.
                user_rows = _fetch_user_experiences(conn, school_id, min(limit, 8))
            except Exception as e:
                print(f"[ApplicantSearch] Querying the user-shared experiences failed; skipping: {e}")
                conn.rollback()     # otherwise the external query below fails on the aborted transaction

        rows: list[dict] = []
        external_limit = 0 if user_submissions_only(query) else max(limit - len(user_rows), 0)
        try:
            # First priority: FTS keyword retrieval (when there is a tsquery)
            if tsquery_str and external_limit > 0:
                sql = f"""
                    SELECT {_COLS},
                           ts_rank(fts_vector, to_tsquery('simple', %(q)s)) AS rank
                    FROM applicant_reports
                    WHERE fts_vector @@ to_tsquery('simple', %(q)s)
                """
                params: dict = {"q": tsquery_str, "limit": external_limit}
                if school_id:
                    sql += " AND school_id = %(school_id)s"
                    params["school_id"] = school_id
                sql += " ORDER BY rank DESC LIMIT %(limit)s"
                rows = fetch_dicts(conn, sql, params)

            # Fallback: no keyword match (or no keywords at all) but a school is pinned -> return that school's experience reports
            # (a generic Chinese question like "this school's past application experiences" lands here)
            if not rows and school_id and external_limit > 0:
                rows = fetch_dicts(
                    conn,
                    f"SELECT {_COLS} FROM applicant_reports "
                    "WHERE school_id = %(school_id)s ORDER BY id DESC LIMIT %(limit)s",
                    {"school_id": school_id, "limit": external_limit},
                )
        except Exception as e:
            print(f"[ApplicantSearch] Querying the external application reports failed; skipping: {e}")

    docs = []
    for row in user_rows:
        text = _format_user_experience_text(row)
        if not text:
            continue
        docs.append({
            "type": "applicant_experience",
            "chunk_text": text,
            "source": "user_submission",
            "source_url": None,
            "school_id": school_id or _canonical_user_school_id(row.get("apply_school") or ""),
            "user_experience_id": row.get("id"),
        })

    for row in rows:
        text = _format_report_text(row)
        if not text:
            continue
        docs.append({
            "type":        "applicant_experience",
            "chunk_text":  text,
            "source":      row.get("source"),
            "source_url":  row.get("source_url"),
            "school_id":   row.get("school_id") or "",
        })
    return docs[:limit]
