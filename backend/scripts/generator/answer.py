"""Synchronous and streaming answer generation."""

from __future__ import annotations

from .client import ANSWER_MODEL, get_openai_client
from .context import clean_answer_text
from .prompts import _build_prompt

def generate_answer_stream(query: str, context_docs: list[dict], model_name: str = ANSWER_MODEL, recommendation: bool = False):
    """Streaming version: yield the raw text chunk by chunk. Raises an exception if the API fails."""
    client = get_openai_client()
    prompt = _build_prompt(query, context_docs, recommendation=recommendation)

    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        yield delta or ""


def generate_answer(query: str, context_docs: list[dict], model_name: str = ANSWER_MODEL, recommendation: bool = False) -> str | None:
    """Generate an answer from the retrieved structured data."""
    try:
        client = get_openai_client()
        prompt = _build_prompt(query, context_docs, recommendation=recommendation)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return clean_answer_text(response.choices[0].message.content or "")
    except Exception as e:
        print(f"[OpenAI] Error while generating the answer: {e}")
        return None
