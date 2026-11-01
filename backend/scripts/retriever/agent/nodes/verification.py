"""Retrieval verification, fallback routing, and query refinement."""

from __future__ import annotations

from generator.context import format_context_for_prompt

from ..prompts import _build_refine_prompt, _build_verify_prompt
from ..state import AgentState, _check_cancel, _emit
from .common import (
    _call_llm,
    _deduplicate_docs,
    _llm_is_unavailable,
    _parse_json_object,
    _parse_professor_query,
)

def verifier_node(state: AgentState) -> dict:
    """
    Before the answer is generated, check whether the data collected by search + extension_function really relates to the question and is enough to answer it.
    It only judges good or bad, and neither retries nor re-queries: when the data is off-topic the finalizer tells the user honestly.
    """
    query = state["original_query"]

    search_docs     = state.get("collected_docs", [])
    fulltext_docs   = state.get("fulltext_docs", [])
    extension_docs  = state.get("extension_docs", [])
    experience_docs = state.get("experience_docs", [])
    recommend_docs  = state.get("recommend_docs", [])

    all_docs = _deduplicate_docs(
        extension_docs + search_docs + experience_docs + recommend_docs + fulltext_docs
    )

    if not all_docs:
        return {"verified_docs": [], "is_sufficient": False, "insufficiency_reason": ""}

    # Short-circuit for experience questions: when needs_experience is set and experience reports were really found, let them through.
    # The Verifier's "was the information point touched" check is designed for official fields and does not suit experience questions
    # with no standard answer (the cases are the answer), where review would only misjudge them as insufficient. The guardrail is the generator's unofficial-source warning.
    if state.get("needs_experience", False) and experience_docs:
        print(f"[Verifier] needs_experience is set and {len(experience_docs)} experience reports exist; letting them through")
        return {"verified_docs": all_docs, "is_sufficient": True, "insufficiency_reason": ""}

    if state.get("wants_recommendation", False) and recommend_docs:
        print(f"[Verifier] wants_recommendation is set and {len(recommend_docs)} recommendations exist; letting them through")
        return {"verified_docs": all_docs, "is_sufficient": True, "insufficiency_reason": ""}

    _check_cancel()
    context_text = format_context_for_prompt(all_docs)

    if _llm_is_unavailable():
        print("[Verifier] LLM quota unavailable; using the retrieved data as is")
        return {
            "verified_docs": all_docs,
            "is_sufficient": True,
            "insufficiency_reason": "",
        }

    try:
        parsed = _parse_json_object(_call_llm(_build_verify_prompt(query, context_text)))
        is_sufficient = bool(parsed.get("sufficient", True))
        reason        = str(parsed.get("reason", "")).strip()
    except Exception as e:
        print(f"[Verifier] Judgment failed ({e}); treating the data as sufficient and leaving the check to the generation step")
        is_sufficient = True
        reason        = ""

    print(f"[Verifier] is_sufficient={is_sufficient}" + (f"  reason={reason}" if reason else ""))

    return {
        "verified_docs":        all_docs,
        "is_sufficient":        is_sufficient,
        "insufficiency_reason": reason,
    }


def after_verify(state: AgentState) -> str:
    """
    Route after the Verifier finishes (tiered fallback: SQL -> full-text search -> rewrite and re-query):
    - is_sufficient=True → finalize
    - is_sufficient=False and "no full-text search yet this round" -> fulltext (search document_chunks, then return to verify)
    - is_sufficient=False and "full-text search already done" and data was retrieved and no retry yet -> refine (rewrite the query and run SQL again)
    - no data at all, or still insufficient after a retry -> finalize (say so honestly)
    """
    if state.get("is_sufficient", True):
        return "finalize"

    # First fallback layer: when SQL is not enough, widen to full-text search first (once per round)
    if not state.get("fulltext_done", False):
        return "fulltext"

    # Second fallback layer: full-text is still not enough, data was retrieved and no retry yet, so rewrite the query and run again
    if state.get("verified_docs") and state.get("retry_count", 0) < 1:
        return "refine"

    return "finalize"


# --- Node 4: Refiner --------------------------------------------------------------



def refiner_node(state: AgentState) -> dict:
    """
    When the Verifier finds the data insufficient, rewrite the sub-questions / professor query using insufficiency_reason
    and let route_to_retrieval run search / extension_function again. Runs at most once.
    """
    query           = state["original_query"]
    reason          = state.get("insufficiency_reason", "")
    sub_queries     = state.get("sub_queries", [])
    professor_query = state.get("professor_query")
    retry_count     = state.get("retry_count", 0)

    print(f"\n[Refiner] Data insufficient ({reason or 'no specific reason'}); regenerating the queries (retry {retry_count + 1})")
    _emit({"type": "thinking", "step": "refine"})

    try:
        parsed = _parse_json_object(_call_llm(_build_refine_prompt(query, reason, sub_queries, professor_query)))

        new_sub_queries = [
            str(q).strip() for q in parsed.get("sub_queries", []) if str(q).strip()
        ] or sub_queries

        new_professor_query = _parse_professor_query(parsed.get("professor_query")) or professor_query
    except Exception as e:
        print(f"[Refiner] Parsing failed ({e}); retrying with the original queries")
        new_sub_queries     = sub_queries
        new_professor_query = professor_query

    for i, q in enumerate(new_sub_queries, 1):
        print(f"  Q{i}: {q}")

    return {
        "sub_queries":     new_sub_queries,
        "professor_query": new_professor_query,
        "retry_count":     retry_count + 1,
        "fulltext_docs":   [],      # clear the previous round's full-text results
        "fulltext_done":   False,   # let the rewritten queries run the full-text search once more
    }
