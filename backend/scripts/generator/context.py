"""Formatting helpers for answer text and retrieved context."""

from __future__ import annotations

import re

def clean_answer_text(text: str) -> str:
    """Remove ** bold markers and leftover <span> tags from an answer (keeping the Markdown links inside)."""
    text = text.replace("**", "")
    text = re.sub(
        r"<span[^>]*>\s*(\[[^\]]+\]\([^\)]+\))\s*</span>",
        r"\1", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"</?span[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def format_context_for_prompt(context_docs: list[dict]) -> str:
    """
    Format SQL query results (structured dicts) and professor extension data into a string the LLM reads easily.

    Each doc is a flat dict and can be one of:
      - structured school requirements: {school_id, university_name, source_url, <requirement fields>...}
        (every non-None field other than school_id/university_name/source_url is shown as a requirement)
      - professor data (chunk format): {chunk_text, source_url, school_id, ...}
    """
    formatted_docs = []
    sources_list = []

    for i, doc in enumerate(context_docs):
        sid = doc.get("school_id", "")
        url = doc.get("source_url", "")

        if doc.get("type") == "applicant_experience":
            # Application experience reports (unofficial, individual cases), from this site and from external sources
            formatted_docs.append(
                f"【申請經驗回報（非官方，個別案例）】\n{doc['chunk_text'].strip()}"
            )
            if url:
                sources_list.append({"school": sid, "type": "applicant_experience", "url": url})
        elif doc.get("type") == "recommendation":
            # School recommendations (reach/match/safety tiers); the header is explicit so they are not mistaken for professor data
            formatted_docs.append(f"【選校推薦】\n{doc['chunk_text'].strip()}")
        elif doc.get("chunk_text"):
            # Professor data (kept in plain-text format)
            univ = doc.get("university_name", sid.upper() if sid else "未知學校")
            formatted_docs.append(f"【{univ} 教授資料】\n{doc['chunk_text'].strip()}")
            if url:
                sources_list.append({"school": sid, "type": "professor", "url": url})
        else:
            # Structured SQL results ("query" is an internal sub-question tag, not a requirement, and is not shown to the LLM)
            univ = doc.get("university_name", sid.upper() if sid else "未知學校")
            fields = {k: v for k, v in doc.items()
                      if k not in ("school_id", "university_name", "source_url", "query") and v is not None}
            field_lines = "\n".join(f"  - {k}: {v}" for k, v in fields.items())
            formatted_docs.append(f"【{univ} 申請要求】\n{field_lines}")
            if url:
                sources_list.append({"school": sid, "type": "requirements", "url": url})

    formatted_text = "\n\n".join(formatted_docs)

    sources_section = "\n\n--- 來源列表（在答案中用 Markdown 連結引用） ---\n"
    for s in sources_list:
        sources_section += f"[{s['school']} - {s['type']}]({s['url']})\n"

    return formatted_text + sources_section
