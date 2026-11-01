"""OpenAI client construction and generic LLM calls."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Same as db/connection.py: load backend/.env explicitly --
# a bare load_dotenv() only looks for a .env in the CWD (the project root has none),
# and it used to work only as a side effect of db.connection being imported first.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_BACKEND_ROOT / ".env")

_client = None
_llm_unavailable_reason: str | None = None
_llm_state_lock = threading.Lock()


class LLMUnavailableError(RuntimeError):
    pass


_QUOTA_FALLBACK_MESSAGE = "OpenAI quota exhausted; local database fallback is active"


def llm_is_unavailable() -> bool:
    """Return whether this process has already seen a permanent quota error."""
    return _llm_unavailable_reason is not None


def _is_permanent_quota_error(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in (
        'insufficient_quota',
        'exceeded your current quota',
        'check your plan and billing',
    ))


def _record_llm_failure(exc: Exception) -> bool:
    global _llm_unavailable_reason
    message = str(exc)
    if not _is_permanent_quota_error(message):
        return False
    with _llm_state_lock:
        if _llm_unavailable_reason is None:
            _llm_unavailable_reason = message
            print('[OpenAI] API quota exhausted (429); switched to local database fallback.')
    return True


def _raise_if_llm_unavailable() -> None:
    if _llm_unavailable_reason is not None:
        raise LLMUnavailableError(_QUOTA_FALLBACK_MESSAGE)

# Tiered models:
#   DEFAULT_MODEL -- judgment and structured tasks (decomposer intent, sub-question splitting, verifier, critic,
#                    refiner, text-to-SQL). They only output a JSON judgment, so a cheap, fast mini model is enough.
#   ANSWER_MODEL  -- final answer generation (the long, user-facing text), worth a stronger model.
# Both can be overridden through environment variables without changing code.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o")


def _sanitize_ssl_env() -> None:
    """Ignore broken SSL env vars that point to non-existent cert paths."""
    cert_file = os.getenv("SSL_CERT_FILE")
    if cert_file and not Path(cert_file).exists():
        print(f"[openai] SSL_CERT_FILE is invalid and was ignored: {cert_file}")
        os.environ.pop("SSL_CERT_FILE", None)

    cert_dir = os.getenv("SSL_CERT_DIR")
    if cert_dir and not Path(cert_dir).exists():
        print(f"[openai] SSL_CERT_DIR is invalid and was ignored: {cert_dir}")
        os.environ.pop("SSL_CERT_DIR", None)


def get_openai_client() -> OpenAI:
    """Get the OpenAI client (singleton)."""
    global _client
    _raise_if_llm_unavailable()
    if _client is None:
        _sanitize_ssl_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY was not found in .env")
        kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        _client = OpenAI(
            **kwargs,
            max_retries=int(os.getenv('LLM_SDK_MAX_RETRIES', '0')),
            timeout=float(os.getenv('LLM_REQUEST_TIMEOUT_SECONDS', '45')),
        )
    return _client


def call_llm(prompt: str, model_name: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Call OpenAI Chat Completions and return plain text.

    Default temperature=0: judgment tasks such as the decomposer, verifier, critic and text-to-SQL
    need stable, reproducible output, so the same question is not sometimes accepted and sometimes refused.
    """
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception as exc:
        if _record_llm_failure(exc):
            raise LLMUnavailableError(_QUOTA_FALLBACK_MESSAGE) from None
        raise
    return (response.choices[0].message.content or "").strip()
