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


def is_citable_url(url: object) -> bool:
    """Whether a source can be shown as a link. Community reports without a page use dedup keys such as
    "gradcafe:123" as their source_url, and those must not be offered to the language model as links."""
    return isinstance(url, str) and url.startswith(("http://", "https://"))


def _doc_urls(doc: dict) -> list[str]:
    """Every citable source of a doc. Chunks have source_url; SQL rows on programs have the source_urls array."""
    urls = [doc.get("source_url")]
    if isinstance(doc.get("source_urls"), list):
        urls += doc["source_urls"]
    return [url for url in dict.fromkeys(urls) if is_citable_url(url)]


def format_context_for_prompt(context_docs: list[dict]) -> str:
    """
    Format the retrieved documents into a string the LLM reads easily. Each one is labeled by the kind of
    source it came from, because the prompts tell official requirements and community reports apart by that label.

    Each doc is a flat dict and can be one of:
      - structured school requirements: {school_id, university_name, source_urls, <requirement fields>...}
        (every non-None field other than the identifying ones is shown as a requirement)
      - official page passage: {type: "official_page", chunk_text, source_url, school_id, university_name}
      - community report: {type: "applicant_experience", chunk_text, source_url, school_id}
      - school recommendation: {type: "recommendation", chunk_text, school_id}
      - faculty data: {chunk_text, source_url, school_id}
    """
    formatted_docs = []
    sources_list = []

    def add_sources(doc: dict, school: str, source_type: str) -> None:
        for url in _doc_urls(doc):
            source = {"school": school, "type": source_type, "url": url}
            if source not in sources_list:
                sources_list.append(source)

    for doc in context_docs:
        sid = doc.get("school_id") or ""
        univ = doc.get("university_name") or (sid.upper() if sid else "未知學校")
        doc_type = doc.get("type")

        if doc_type == "applicant_experience":
            # Application experience reports (unofficial, individual cases), from this site and from external sources
            formatted_docs.append(
                f"【申請經驗回報（非官方，個別案例）】\n{doc['chunk_text'].strip()}"
            )
            add_sources(doc, sid, "applicant_experience")
        elif doc_type == "recommendation":
            # School recommendations (reach/match/safety tiers); the header is explicit so they are not mistaken for faculty data
            formatted_docs.append(f"【選校推薦】\n{doc['chunk_text'].strip()}")
        elif doc_type == "official_page":
            # Full-text and hybrid search results: passages from the crawled official pages
            formatted_docs.append(f"【{univ} 官方頁面內容】\n{doc['chunk_text'].strip()}")
            add_sources(doc, sid, "official_page")
        elif doc.get("chunk_text"):
            # Faculty data (kept in plain-text format)
            formatted_docs.append(f"【{univ} 教授資料】\n{doc['chunk_text'].strip()}")
            add_sources(doc, sid, "professor")
        else:
            # Structured SQL results ("query" is an internal sub-question tag, not a requirement, and is not shown to the LLM)
            fields = {k: v for k, v in doc.items()
                      if k not in ("school_id", "university_name", "source_url", "source_urls", "query") and v is not None}
            field_lines = "\n".join(f"  - {k}: {v}" for k, v in fields.items())
            formatted_docs.append(f"【{univ} 申請要求】\n{field_lines}")
            add_sources(doc, sid, "requirements")

    formatted_text = "\n\n".join(formatted_docs)

    sources_section = "\n\n--- 來源列表（在答案中用 Markdown 連結引用） ---\n"
    for s in sources_list:
        sources_section += f"[{s['school']} - {s['type']}]({s['url']})\n"

    return formatted_text + sources_section
