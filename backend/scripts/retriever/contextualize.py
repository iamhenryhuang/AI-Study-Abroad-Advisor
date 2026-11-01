"""Rewrite a follow-up question that depends on earlier turns into a standalone question (the first step of multi-turn chat).

It sits before the agent: the agent itself still takes a single query, and the multi-turn context is
collapsed here into one standalone question. When the LLM fails it falls back to the original query so the rewrite never blocks the conversation.
"""
from __future__ import annotations

from generator.client import call_llm


def _build_prompt(query: str, history: list[dict]) -> str:
    lines = []
    for m in history:
        role = "使用者" if m.get("role") == "user" else "助理"
        lines.append(f"{role}：{m.get('content', '')}")
    convo = "\n".join(lines)
    return f"""你是一個問題改寫助理。根據以下對話歷史，把使用者最新的「跟隨問題」改寫成一個不需要上下文也能獨立理解的完整問題。

規則：
- 把代名詞（它、那個、這所學校…）換成歷史中明確的對象。
- 若最新問題本身已是獨立問題，原樣輸出即可。
- 只輸出改寫後的問題，不要任何解釋或前綴。

【對話歷史】
{convo}

【最新跟隨問題】
{query}

【改寫後的獨立問題】"""


def contextualize_query(query: str, history: list[dict]) -> str:
    """When there is history, use the LLM to rewrite the follow-up into a standalone question; with no history return the query as is.

    history: [{"role": "user"|"assistant", "content": str}, ...]
    When the rewrite is empty or the LLM call fails, fall back to the original query.
    """
    if not history:
        return query
    try:
        rewritten = call_llm(_build_prompt(query, history))
        return rewritten.strip() or query
    except Exception as e:
        print(f"[contextualize] Rewrite failed; using the original question: {e}")
        return query
