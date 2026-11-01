"""Final answer generation and hallucination critic nodes."""

from __future__ import annotations

import re

from langgraph.graph import END

from generator.answer import generate_answer, generate_answer_stream
from generator.client import llm_is_unavailable
from generator.context import clean_answer_text, format_context_for_prompt
from retriever.sql_search import get_known_school_ids

from ..prompts import _build_critic_prompt
from ..state import AgentState, _check_cancel, _emit
from .common import _call_llm, _parse_json_object


def _extractive_fallback_answer(docs: list[dict]) -> str:
    types = {doc.get('type') for doc in docs}
    professor_mode = 'professor_list' in types
    experience_mode = 'applicant_experience' in types
    recommendation_mode = 'recommendation' in types

    if professor_mode:
        title = '教授名單（直接取自資料庫）'
        limit = 30
    elif experience_mode:
        title = '申請經驗回報（本站分享與外部案例；非官方、僅供參考）'
        limit = 12
    elif recommendation_mode:
        title = '選校分級結果（直接取自資料庫統計）'
        limit = 12
    else:
        title = '官方頁面全文檢索結果'
        limit = 5

    lines = [
        f'### {title}',
        '',
        '目前 LLM 額度不可用，以下內容未經生成式模型改寫，直接摘錄自檢索資料。',
    ]

    seen = set()
    index = 0
    for doc in docs:
        text = str(doc.get('chunk_text') or '').strip()
        if not text:
            fields = [
                f'{key}: {value}'
                for key, value in doc.items()
                if key not in {'query', 'source_url'} and value not in (None, '', [], {})
            ]
            text = '；'.join(fields)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        index += 1
        if len(text) > 1000:
            text = text[:1000].rstrip() + '…'
        school = str(doc.get('university_name') or doc.get('school_id') or '').upper()
        prefix = f'[{school}] ' if school else ''
        lines.extend(['', f'{index}. {prefix}{text}'])
        url = str(doc.get('source_url') or '').strip()
        if url:
            lines.append(f'   來源：[{url}]({url})')
        if index >= limit:
            break

    if index == 0:
        return '很抱歉，資料庫中沒有可直接呈現的相關內容。'
    return '\n'.join(lines)

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
        mentioned_ids   = state.get("mentioned_school_ids", [])
        mentioned_names = state.get("mentioned_school_names", [])
        known = get_known_school_ids()

        # mentioned_school_ids and mentioned_school_names are two independent detection results and can both be present
        # (for example a question about a known MIT and an unlisted University of Toronto), so they are judged separately, not exclusively.
        # school_ids is limited to the known alias list, which is kept in sync with what the DB covers, so it is always in known;
        # whether a school is covered really depends on mentioned_school_names (the original text, including unlisted schools).
        unknown_ids   = [sid for sid in mentioned_ids if sid not in known]
        unrecognized = unknown_ids + mentioned_names

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
        fallback = _extractive_fallback_answer(all_docs)
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
        fallback = _extractive_fallback_answer(all_docs)
        print('[Finalizer] LLM unavailable; returning grounded extractive fallback')
        _emit({'type': 'answer', 'text': fallback})
        return {'final_answer': fallback, 'generated_answer': False}

    if full_text:
        clean = clean_answer_text(full_text)
        return {"final_answer": clean, "generated_answer": True}
    else:
        msg = "OpenAI generation failed"
        print(f"[Finalizer] {msg}")
        _emit({"type": "error", "message": msg})
        return {"final_answer": ""}


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
