"""Answer generation: streaming, non-streaming, and extractive without a language model."""

from __future__ import annotations

import re

from .client import ANSWER_MODEL, get_openai_client, record_llm_failure
from .context import clean_answer_text, is_citable_url
from .prompts import build_prompt


def generate_answer_stream(query: str, context_docs: list[dict], model_name: str = ANSWER_MODEL, recommendation: bool = False):
    """Streaming version: yield the raw text chunk by chunk. Raises an exception if the API fails."""
    client = get_openai_client()
    prompt = build_prompt(query, context_docs, recommendation=recommendation)

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            yield delta or ""
    except Exception as exc:
        record_llm_failure(exc)
        raise


def generate_answer(query: str, context_docs: list[dict], model_name: str = ANSWER_MODEL, recommendation: bool = False) -> str | None:
    """Generate an answer from the retrieved structured data."""
    try:
        client = get_openai_client()
        prompt = build_prompt(query, context_docs, recommendation=recommendation)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return clean_answer_text(response.choices[0].message.content or "")
    except Exception as e:
        record_llm_failure(e)
        print(f"[OpenAI] Error while generating the answer: {e}")
        return None


def extractive_answer(docs: list[dict]) -> str:
    """Quote the retrieved documents directly, for when no language model is available."""
    types = {doc.get('type') for doc in docs}

    if 'professor_list' in types:
        title = '教授名單（直接取自資料庫）'
        limit = 30
    elif 'applicant_experience' in types:
        title = '申請經驗回報（本站分享與外部案例；非官方、僅供參考）'
        limit = 12
    elif 'recommendation' in types:
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
                if key not in {'query', 'source_url', 'source_urls'} and value not in (None, '', [], {})
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
        if is_citable_url(url):
            lines.append(f'   來源：[{url}]({url})')
        if index >= limit:
            break

    if index == 0:
        return '很抱歉，資料庫中沒有可直接呈現的相關內容。'
    return '\n'.join(lines)
