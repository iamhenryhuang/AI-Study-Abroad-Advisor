"""Shared helpers for agent node implementations."""

from __future__ import annotations

import hashlib
import json
import re

from generator.client import call_llm, llm_is_unavailable

from ..state import _SCHOOL_ALIASES, _check_cancel, _detect_school_ids

def _deduplicate_docs(docs: list[dict]) -> list[dict]:
    """Remove duplicate documents by school_id + chunk_text / full field content.

    For structured SQL docs the hash of the full content is the key (no truncation), so two docs that
    differ only in the trailing fields (such as application_url) are not mistaken for duplicates by a prefix cut.
    The "query" field (the sub-question tag injected by _search_one_query) is left out of the hash --
    otherwise two sub-questions that return the same row would be treated as different because their query differs.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for doc in docs:
        content = doc.get("chunk_text") or json.dumps(
            {k: v for k, v in doc.items() if k != "query"}, sort_keys=True, default=str)
        key = f"{doc.get('school_id', '')}:{hashlib.md5(content.encode('utf-8')).hexdigest()}"
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def _call_llm(prompt: str) -> str:
    """Call OpenAI and return plain text. Checks for cancellation before every call."""
    _check_cancel()
    return call_llm(prompt)


def _llm_is_unavailable() -> bool:
    return llm_is_unavailable()


def _parse_json_object(raw: str) -> dict:
    """Extract the first JSON object from an LLM reply and parse it; raise on failure."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())


def _parse_professor_query(pq_raw: object) -> dict | None:
    """
    Normalize the professor_query object returned by the LLM into {name, school, school_id}.
    When school_id is not in the known alias table, try to re-detect it from the school name; if still not found, keep the original string.
    decomposer_node and refiner_node share this logic so neither keeps its own copy.
    """
    if not isinstance(pq_raw, dict):
        return None
    pq_name = (pq_raw.get("name") or "").strip()
    if not pq_name:
        return None
    pq_school = (pq_raw.get("school") or "").strip()
    pq_sid    = (pq_raw.get("school_id") or "").strip()
    if pq_sid not in _SCHOOL_ALIASES:
        detected = _detect_school_ids(pq_school)
        pq_sid = detected[0] if detected else (pq_school or pq_sid)
    return {"name": pq_name, "school": pq_school, "school_id": pq_sid}
