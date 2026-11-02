"""
Agent interface layer: exposes the fetch_professor_docs() function.

The decomposer has already judged the intent and stored it in state;
this module only calls SerpAPI for professor data and converts it to chunk format.
"""

from __future__ import annotations

import time

from professor_fetcher.fetcher import (
    search_professor_id,
    fetch_author_profile_and_recent_papers,
)
from professor_fetcher.formatter import format_professor_to_json


def fetch_one(
    name: str,
    school: str,
    school_id: str,
    author_id: str = "",
    cutoff_year: int | None = None,
    max_papers: int = 20,
    delay: float = 1.0,
) -> list[dict] | None:
    """Run the single-professor fetch flow and return a list of records in the new format."""
    print(f"[ProfessorFetcher] Processing: {name} @ {school}")

    if not author_id:
        author_id = search_professor_id(name, affiliation=school)
        time.sleep(delay)

    if not author_id:
        print(f"!!! Skipping {name}: Author ID not found.")
        return None

    profile, papers = fetch_author_profile_and_recent_papers(
        author_id=author_id,
        professor_name=name,
        cutoff_year=cutoff_year,
        max_papers=max_papers,
    )
    time.sleep(delay)

    return format_professor_to_json(
        profile_data=profile,
        recent_papers=papers,
        school_id=school_id,
        school_name=school,
        professor_name=name,
        author_id=author_id,
    )


# --- Public interface -------------------------------------------------------------

def _fetch_professor_docs(name: str, school: str, school_id: str, query: str) -> list[dict]:
    """Fetch the named professor through SerpAPI and convert the result to chunk format."""
    print(f"[ProfessorFetcher] Fetching: {name} @ {school} ({school_id})")

    try:
        records = fetch_one(name=name, school=school, school_id=school_id)
    except Exception as e:
        print(f"[ProfessorFetcher] Fetch failed: {e}")
        return []

    if not records:
        print("[ProfessorFetcher] No data returned")
        return []

    docs = []
    for rec in records:
        text = rec.get("data", "").strip()
        if not text:
            continue
        docs.append({
            "chunk_text":   text,
            "source_url":   rec.get("url", ""),
            "passed_types": rec.get("passed_types", []),
            "school_id":    rec.get("school_id", ""),
            "rerank_score": 0.8,
            "query":        query,
        })

    print(f"[ProfessorFetcher] Got {len(docs)} chunk documents")
    return docs


def run_professor_fetch(professor_query: dict | None, query: str) -> list[dict]:
    """
    Single entry point of the extension function node.
    Takes the professor_query parsed by the decomposer and calls SerpAPI for the data.

    Args:
        professor_query: {"name": str, "school": str, "school_id": str} or None
        query:           the user's original question

    Returns:
        A list of chunk-format docs; an empty list when there is no professor query or it fails.
    """
    if not professor_query:
        return []

    return _fetch_professor_docs(
        name=professor_query["name"],
        school=professor_query["school"],
        school_id=professor_query["school_id"],
        query=query,
    )
