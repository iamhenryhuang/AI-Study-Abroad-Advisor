"""Professor, experience, SQL, and full-text retrieval nodes."""

from __future__ import annotations

from professor_fetcher.fetch_for_agent import run_professor_fetch
from retriever.applicant_search import applicant_search
from retriever.hybrid_search import hybrid_search_with_fallback
from retriever.professor_list_search import professor_list_search
from retriever.sql_search import sql_search
from retriever.recommend import recommend, fetch_nearby_cases

from ..state import AgentState, _check_cancel, _detect_school_ids, _emit

def extension_function_node(state: AgentState) -> dict:
    """Extension node (runs in parallel with search): fetch professor data and write extension_docs.

    Handles two mutually exclusive kinds of professor query:
    - professor_query: a named professor; use SerpAPI/Google Scholar for the research details
    - professor_list_query: only a school; query the professors table for that school's professor list
    """
    _emit({"type": "thinking", "step": "extension_function"})

    extension_docs: list[dict] = []

    professor_query = state.get("professor_query")
    if professor_query is not None:
        print(f"[Extension] Fetching professor data: {professor_query.get('name')}")
        docs = run_professor_fetch(professor_query, state["original_query"])
        if docs:
            _emit({
                "type": "tool_result",
                "tool": "fetch_professor",
                "preview": f"Found {len(docs)} professor-related items",
            })
            extension_docs.extend(docs)
        else:
            print("[Extension] No professor data fetched")

    professor_list_query = state.get("professor_list_query")
    if professor_list_query is not None:
        school_id = professor_list_query.get("school_id")
        print(f"[Extension] Querying the professor list: {school_id}")
        _emit({
            "type": "tool_call",
            "tool": "professor_list_search",
            "args": {"school_id": school_id},
        })
        docs = professor_list_search(school_id)
        _emit({
            "type": "tool_result",
            "tool": "professor_list_search",
            "preview": f"Found {len(docs)} professors",
        })
        if docs:
            extension_docs.extend(docs)
        else:
            print(f"[Extension] No professor list data for {school_id}")

    print(f"[Extension] Got {len(extension_docs)} extension items in total")
    return {
        "extension_docs":    extension_docs,
    }


# --- Node 1.6: Experience Search (application experience reports) ----------------

def experience_search_node(state: AgentState) -> dict:
    """Query applicant_reports + user_experiences and write experience_docs.

    Runs in parallel with search / extension_function; the data are unofficial anecdotes and the generator adds a warning.
    """
    _emit({"type": "thinking", "step": "experience_search"})

    original_query = state["original_query"]
    sub_queries    = state.get("sub_queries", []) or [original_query]
    fallback_ids   = _detect_school_ids(original_query)
    fallback_id    = fallback_ids[0] if fallback_ids else None

    exp_docs: list[dict] = []
    seen_urls: set = set()
    for q in sub_queries:
        _check_cancel()
        # Detect the school separately for every sub-question, so a multi-school question (such as "which is easier, A or B")
        # is not overridden by the first school found in the whole sentence, which would leave the other schools with no data.
        q_school_ids = _detect_school_ids(q)
        school_id = q_school_ids[0] if q_school_ids else fallback_id
        _emit({
            "type": "tool_call",
            "tool": "applicant_search",
            "args": {"query": q, **({"school_id": school_id} if school_id else {})},
        })
        docs = applicant_search(q, school_id=school_id)
        for doc in docs:
            key = doc.get("source_url") or doc.get("chunk_text", "")[:80]
            if key not in seen_urls:
                seen_urls.add(key)
                exp_docs.append(doc)

    _emit({
        "type": "tool_result",
        "tool": "applicant_search",
        "preview": f"Found {len(exp_docs)} application experience reports",
    })
    print(f"[Experience] Got {len(exp_docs)} application experience reports in total")

    return {"experience_docs": exp_docs}


# --- Node 2: SQL Searcher ---------------------------------------------------------

def _search_one_query(q: str) -> list[dict]:
    """Run text-to-SQL retrieval for one sub-question (the school filter comes from the WHERE clause the LLM writes)."""
    _emit({
        "type": "tool_call",
        "tool": "sql_search",
        "args": {"query": q},
    })

    results, sql = sql_search(q)

    for item in results:
        item["query"] = q

    _emit({
        "type": "tool_result",
        "tool": "sql_search",
        "preview": f"Found {len(results)} related items" + (f" (SQL: {sql})" if sql else ""),
    })

    return results


def searcher_node(state: AgentState) -> dict:
    """Run text-to-SQL retrieval for all sub-questions in a single pass (no retry loop)."""
    sub_queries = state.get("sub_queries", [])

    print(f"\n[Searcher] {len(sub_queries)} sub-questions in total")
    _emit({"type": "thinking", "step": 1})

    new_docs: list[dict] = []

    for q in sub_queries:
        _check_cancel()
        results = _search_one_query(q)
        print(f"     Query: {q}  got {len(results)} items (SQL)")
        new_docs.extend(results)

    print(f"\n[Searcher] Got {len(new_docs)} items in total")

    return {"collected_docs": new_docs}


# --- Node 2.5: Fulltext Search (fallback when SQL is not enough) ------------------

def _fulltext_one_query(q: str, original_query: str) -> list[dict]:
    """Run a document_chunks full-text search for one sub-question."""
    school_ids = _detect_school_ids(q) or _detect_school_ids(original_query)
    school_id = school_ids[0] if school_ids else None

    _emit({
        "type": "tool_call",
        "tool": "fulltext_search",
        "args": {"query": q, **({"school_id": school_id} if school_id else {})},
    })

    results = hybrid_search_with_fallback(q, school_id=school_id)

    for item in results:
        item["query"] = q

    _emit({
        "type": "tool_result",
        "tool": "fulltext_search",
        "preview": f"Full-text search found {len(results)} related passages",
    })

    return results


def fulltext_search_node(state: AgentState) -> dict:
    """
    Only when the Verifier finds the text-to-SQL results insufficient, add a full-text search over document_chunks.
    This is the second layer of the tiered fallback: first query the structured fields with SQL, and when SQL cannot answer (off-topic or missing fields),
    widen to the original page text, merge the chunks into collected_docs and return to verify for a new judgment.
    """
    sub_queries    = state.get("sub_queries", [])
    original_query = state["original_query"]

    print(f"\n[Fulltext] SQL results were insufficient; running the full-text fallback for {len(sub_queries)} sub-questions")
    _emit({"type": "thinking", "step": "fulltext"})

    ft_docs: list[dict] = []
    for q in sub_queries:
        _check_cancel()
        results = _fulltext_one_query(q, original_query)
        print(f"     Query: {q}  got {len(results)} items (full-text)")
        ft_docs.extend(results)

    print(f"\n[Fulltext] Got {len(ft_docs)} full-text items in total")

    return {"fulltext_docs": ft_docs, "fulltext_done": True}


def recommend_node(state: AgentState) -> dict:
    """Tier schools from the profile (reach/match/safety) with real cases as evidence, and write recommend_docs."""
    _emit({"type": "thinking", "step": "recommend"})
    profile = state.get("profile") or {}
    docs: list[dict] = []
    try:
        tiers = recommend(profile)
        for tier, schools in tiers.items():
            for s in schools:
                cases = fetch_nearby_cases(s["school_id"], profile.get("gpa"))
                case_txt = "；".join(
                    f"GPA {c.get('gpa')} {c.get('decision')}" for c in cases
                ) or "（無相近案例）"
                docs.append({
                    "type":       "recommendation",
                    "school_id":  s["school_id"],
                    "chunk_text": (f"[{tier}] {s['name']}\n"
                                   f"分數對比：{'；'.join(s['comparison'])}\n"
                                   f"相近錄取案例：{case_txt}"),
                    "source_url": "",
                })
    except Exception as e:
        print(f"[Recommend] Recommendation failed: {e}")
        return {"recommend_docs": []}
    _emit({"type": "tool_result", "tool": "recommend",
           "preview": f"Tiered recommendation of {len(docs)} schools"})
    print(f"[Recommend] Produced {len(docs)} recommendations")
    return {"recommend_docs": docs}
