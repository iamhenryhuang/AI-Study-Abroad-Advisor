"""Final answer generation and hallucination critic nodes."""

from __future__ import annotations

from langgraph.graph import END

from generator.answer import extractive_answer, generate_answer, generate_answer_stream
from generator.client import llm_is_unavailable
from generator.context import clean_answer_text, format_context_for_prompt
from retriever.school_aliases import detect_school_ids
from retriever.search.sql_search import get_known_school_ids

from ..prompts import _build_critic_prompt
from ..state import AgentState, _check_cancel, _emit
from .common import _call_llm, _parse_json_object


def _uncovered_schools(mentioned_ids: list[str], mentioned_names: list[str], known: set[str]) -> list[str]:
    """Which of the schools the question mentions have no crawled data.

    The two inputs are independent detection results and can overlap (a known MIT next to an unlisted
    University of Toronto, or "stanford" next to "Stanford University"), so each school is reported at most once.
    """
    uncovered = []
    reported_ids: set[str] = set()
    for name in mentioned_names:
        ids = detect_school_ids(name)
        if ids and all(sid in known for sid in ids):
            continue
        uncovered.append(name)
        reported_ids.update(ids)
    uncovered += [sid for sid in mentioned_ids if sid not in known and sid not in reported_ids]
    return uncovered


def finalizer_node(state: AgentState) -> dict:
    """
    Generate the final answer from the data the Verifier judged.
    If the Verifier finds the data insufficient or off-topic, say so honestly and send the answer event directly (no Critic review needed).
    If an answer is generated, hand it to the Critic to check for hallucination, and the Critic then sends the final answer event.
    """
    query = state["original_query"]
    all_docs = state.get("verified_docs", [])

    print(f"\n[Finalizer] Data after Verifier: {len(all_docs)} items")

    if not all_docs:
        known = get_known_school_ids()
        # known is None when the database could not be reached; then nothing can be said about coverage.
        unrecognized = [] if known is None else _uncovered_schools(
            state.get("mentioned_school_ids", []), state.get("mentioned_school_names", []), known,
        )

        if unrecognized:
            names = "、".join(unrecognized)
            answer = f"很抱歉，目前系統尚未收錄 {names} 的資料，暫時無法回答此問題。建議您直接前往該校官方網站查詢。"
        else:
            answer = "很抱歉，資料庫中查無與此問題相關的欄位資訊。建議您直接前往各校官方網站查詢。"

        _emit({"type": "answer", "text": answer})
        return {"final_answer": answer}

    if not state.get("is_sufficient", True):
        reason = state.get("insufficiency_reason", "").strip()
        answer = "很抱歉，目前檢索到的資料與您的問題不符" + (f"（{reason}）" if reason else "") + "，暫時無法提供可靠的回答。建議您直接前往官方網站查詢。"
        _emit({"type": "answer", "text": answer})
        return {"final_answer": answer}

    if llm_is_unavailable():
        fallback = extractive_answer(all_docs)
        print('[Finalizer] LLM quota unavailable; returning database excerpts directly')
        _emit({'type': 'answer', 'text': fallback})
        return {'final_answer': fallback, 'generated_answer': False}

    _check_cancel()
    _emit({"type": "llm_call", "purpose": "finalizer"})

    recommendation = state.get("wants_recommendation", False)
    full_text = ""
    try:
        for chunk in generate_answer_stream(query, all_docs, recommendation=recommendation):
            _check_cancel()
            full_text += chunk
            if chunk:
                _emit({"type": "answer_chunk", "text": chunk})
    except Exception as e:
        print(f"[Finalizer] Streaming failed, falling back to non-streaming: {e}")
        try:
            full_text = generate_answer(query, all_docs, recommendation=recommendation) or ""
        except Exception as fallback_error:
            print(f"[Finalizer] Non-streaming is also unavailable, using database excerpts: {fallback_error}")
            full_text = ""

    if not full_text:
        fallback = extractive_answer(all_docs)
        print('[Finalizer] LLM unavailable; returning grounded extractive fallback')
        _emit({'type': 'answer', 'text': fallback})
        return {'final_answer': fallback, 'generated_answer': False}

    return {"final_answer": clean_answer_text(full_text), "generated_answer": True}


def after_finalize(state: AgentState) -> str:
    """Run the Critic review only when the LLM actually generated an answer; the other cases (honest refusal / generation failure) end here."""
    if state.get("generated_answer", False):
        return "critic"
    return END


# --- Node 6: Critic ---------------------------------------------------------------

def critic_node(state: AgentState) -> dict:
    """
    After the answer is generated and before it goes back to the user, check whether any content has no support in verified_docs (hallucination).
    It only flags a warning and never regenerates: when a problem is found the warning is appended to the answer for the user to confirm.
    """
    answer   = state.get("final_answer", "") or ""
    all_docs = state.get("verified_docs", [])

    if not answer or not all_docs:
        _emit({"type": "answer", "text": answer})
        return {"final_answer": answer}

    _check_cancel()
    context_text = format_context_for_prompt(all_docs)

    try:
        parsed        = _parse_json_object(_call_llm(_build_critic_prompt(answer, context_text)))
        has_issue     = bool(parsed.get("has_issue", False))
        issue_summary = str(parsed.get("issue_summary", "")).strip()
    except Exception as e:
        print(f"[Critic] Judgment failed ({e}); skipping the quality check")
        has_issue     = False
        issue_summary = ""

    if has_issue:
        print(f"[Critic] Found a potential problem: {issue_summary}")
        final_text = answer + f"\n\n⚠️ 提醒：{issue_summary or '部分內容可能未完全對應參考資料，建議自行確認。'}"
    else:
        print("[Critic] No hallucination or off-topic problem found")
        final_text = answer

    _emit({"type": "answer", "text": final_text})
    return {"final_answer": final_text}
