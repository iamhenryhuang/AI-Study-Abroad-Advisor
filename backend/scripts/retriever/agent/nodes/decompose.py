"""Query decomposition and retrieval routing nodes."""

from __future__ import annotations

import re

from langgraph.types import Send

from ..prompts import _build_intent_prompt, _build_subquery_prompt
from ..state import AgentState, _SCHOOL_ALIASES, _detect_school_ids
from .common import _call_llm, _llm_is_unavailable, _parse_json_object, _parse_professor_query


def _extract_profile(query: str) -> dict:
    low = query.lower()
    profile = {}
    patterns = {
        'gpa': r'\bgpa\s*[:：]?\s*(\d+(?:\.\d+)?)',
        'toefl': r'\btoefl\s*[:：]?\s*(\d+(?:\.\d+)?)',
        'ielts': r'\bielts\s*[:：]?\s*(\d+(?:\.\d+)?)',
        'gre': r'\bgre\s*[:：]?\s*(\d+(?:\.\d+)?)',
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, low, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            profile[field] = int(value) if value.is_integer() else value
    return profile


def _fallback_intent(query: str) -> dict:
    low = query.lower()
    school_ids = _detect_school_ids(query)
    wants_recommendation = any(term in low for term in (
        '推薦學校', '選校', '衝刺', '保底', 'recommend school', 'school recommendation',
    ))
    needs_experience = any(term in low for term in (
        '錄取者', '錄取背景', '錄取機會', '申請案例', '申請經驗', '分享經驗',
        '使用者分享', '經驗回報', '有沒有機會',
        'admitted profile', 'admission chance', 'acceptance chance', 'application experience',
    ))
    professor_intent = any(term in low for term in (
        '教授', '師資', 'faculty', 'professor', 'researcher',
    ))
    professor_list_query = None
    if professor_intent and school_ids:
        professor_list_query = {'school_id': school_ids[0]}

    profile = _extract_profile(query)
    needs_sql_search = not (
        professor_list_query is not None or needs_experience or wants_recommendation
    )
    return {
        'school_ids': school_ids,
        'mentioned_school_names': [],
        'professor_query': None,
        'professor_list_query': professor_list_query,
        'needs_sql_search': needs_sql_search,
        'needs_experience': needs_experience,
        'wants_recommendation': wants_recommendation,
        'profile': profile,
    }

def decomposer_node(state: AgentState) -> dict:
    query = state["original_query"]
    print(f"\n{'='*60}")
    print(f"[Decomposer] Original question: {query}")
    print(f"{'='*60}")

    # Tasks 1+2+3: recognize the school and professor intent, and decide whether SQL is needed (this decides the routing)
    try:
        parsed = _parse_json_object(_call_llm(_build_intent_prompt(query)))
        school_ids = [
            str(s).strip()
            for s in parsed.get("school_ids", [])
            if str(s).strip() in _SCHOOL_ALIASES
        ]
        mentioned_school_names = [
            str(s).strip() for s in parsed.get("mentioned_school_names", []) if str(s).strip()
        ]
        professor_query  = _parse_professor_query(parsed.get("professor_query"))
        professor_list_query = parsed.get("professor_list_query") or None
        needs_sql_search = bool(parsed.get("needs_sql_search", True))
        needs_experience = bool(parsed.get("needs_experience", False))
        wants_recommendation = bool(parsed.get("wants_recommendation", False))
        profile = parsed.get("profile") or {}
    except Exception as e:
        if _llm_is_unavailable():
            print("[Decomposer] LLM quota unavailable; falling back to local rules for the intent")
        else:
            print(f"[Decomposer] Intent detection failed ({e}); using the original question as the fallback")
        fallback                = _fallback_intent(query)
        school_ids              = fallback['school_ids']
        mentioned_school_names  = fallback['mentioned_school_names']
        professor_query         = fallback['professor_query']
        professor_list_query    = fallback['professor_list_query']
        needs_sql_search        = fallback['needs_sql_search']
        needs_experience        = fallback['needs_experience']
        wants_recommendation    = fallback['wants_recommendation']
        profile                 = fallback['profile']

    # Task 4: split into sub-questions only when SQL is needed (a pure professor query saves this LLM call)
    if needs_sql_search and not _llm_is_unavailable():
        try:
            parsed_sq = _parse_json_object(_call_llm(_build_subquery_prompt(query, school_ids)))
            sub_queries = [str(q).strip() for q in parsed_sq.get("sub_queries", []) if str(q).strip()]
            if not sub_queries:
                raise ValueError("sub_queries is empty")
        except Exception as e:
            print(f"[Decomposer] Sub-question split failed ({e}); using the original question as the fallback")
            sub_queries = [query]
    else:
        sub_queries = [query] if needs_sql_search else []

    if len(school_ids) >= 2:
        print(f"[Decomposer] Detected {len(school_ids)} schools, split into {len(sub_queries)} sub-questions:")
    else:
        label = f"(school: {school_ids[0]})" if school_ids else "(no specific school)"
        print(f"[Decomposer] Single question {label}:")
    for i, q in enumerate(sub_queries, 1):
        print(f"  Q{i}: {q}")

    if professor_query:
        print(f"[Decomposer] professor_query = {professor_query['name']} @ {professor_query['school']} "
              f"[{professor_query.get('school_id', '?')}]")
    else:
        print("[Decomposer] professor_query = None (no professor intent)")
    if professor_list_query:
        print(f"[Decomposer] professor_list_query = {professor_list_query['school_id']}")
    print(f"[Decomposer] needs_sql_search = {needs_sql_search}  needs_experience = {needs_experience}")

    return {
        "sub_queries":            sub_queries,
        "collected_docs":         [],
        "fulltext_docs":          [],
        "fulltext_done":          False,
        "extension_docs":         [],
        "experience_docs":        [],
        "final_answer":           "",
        "professor_query":        professor_query,
        "professor_list_query":   professor_list_query,
        "needs_sql_search":       needs_sql_search,
        "needs_experience":       needs_experience,
        "wants_recommendation":   wants_recommendation,
        "profile":                profile,
        "recommend_docs":         [],
        "mentioned_school_ids":   school_ids,
        "mentioned_school_names": mentioned_school_names,
    }


def route_to_retrieval(state: AgentState):
    """
    Decide which retrieval nodes to run (shared by the route after decompose and the route after refine regenerates the queries):
    The retrieval branches are independent and are triggered in parallel by flags:
    - needs_sql_search -> search (text-to-SQL over official application requirements)
    - professor_query   -> extension_function (SerpAPI professor data)
    - needs_experience  -> experience_search (applicant_reports admission experiences)
    - wants_recommendation -> recommend (tier schools by the user's scores)
    At least one node always runs; if none applies (in theory impossible), it falls back to search alone.
    """
    targets = []
    if state.get("needs_sql_search", True):
        targets.append(Send("search", state))
    if state.get("professor_query") is not None or state.get("professor_list_query") is not None:
        targets.append(Send("extension_function", state))
    if state.get("needs_experience", False):
        targets.append(Send("experience_search", state))
    if state.get("wants_recommendation", False):
        targets.append(Send("recommend", state))

    return targets or [Send("search", state)]

