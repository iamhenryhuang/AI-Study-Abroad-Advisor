"""LLM calls -- reusing the project's existing way of calling and its keys:

- The SDK and interface follow backend/generator/client.py (an openai.OpenAI singleton)
- Keys come from backend/.env; three providers are supported (Groq/Gemini go through OpenAI-compatible endpoints):
    * openai -> OPENAI_API_KEY (model OPENAI_MODEL, default gpt-4.1)
    * groq   -> GROQ_API_KEY (model GROQ_MODEL, default llama-3.3-70b-versatile,
               through its OpenAI-compatible endpoint)
    * gemini -> GOOGLE_API_KEY (model GEMINI_MODEL, default gemini-2.5-flash)
- Selection: the environment variable LLM_PROVIDER=openai|groq|gemini;
  when unset it is detected automatically (openai -> groq -> gemini, the first one that has a key)
- The same SSL env cleanup

This file adds a JSON-mode wrapper (call_llm_json), rate-limit backoff and a concurrency cap.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Same as the backend: load backend/.env (the root .env is tolerated too)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

_client = None
_client_lock = threading.Lock()
_unavailable_reason: str | None = None
_unavailable_lock = threading.Lock()

# Free-tier TPM rate-limit protection: cap the number of simultaneous LLM calls (so a fan-out does not exhaust the quota at once)
_LLM_SEMAPHORE = threading.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "2")))

# Provider settings: key env var / base_url / default model
_PROVIDERS = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        # Official OpenAI by default; set OPENAI_BASE_URL to use a compatible endpoint
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
        "model": lambda: os.getenv("OPENAI_MODEL", "gpt-4.1"),
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
    "gemini": {
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    },
}


def _pick_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit:
        if explicit not in _PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {list(_PROVIDERS)}, got: {explicit}")
        if not os.getenv(_PROVIDERS[explicit]["key_env"]):
            raise ValueError(f"LLM_PROVIDER={explicit} but {_PROVIDERS[explicit]['key_env']} is not set")
        return explicit
    for name in ("openai", "groq", "gemini"):
        if os.getenv(_PROVIDERS[name]["key_env"]):
            return name
    raise ValueError("None of OPENAI_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY was found in .env")


PROVIDER = _pick_provider()
DEFAULT_MODEL = _PROVIDERS[PROVIDER]["model"]()

MAX_LLM_RETRY = 4


class LLMUnavailableError(RuntimeError):
    # The provider cannot serve requests for the remainder of this run.
    pass


_QUOTA_FALLBACK_MESSAGE = "LLM quota exhausted; deterministic crawler fallback is active"


def _is_permanent_quota_error(error_message: str) -> bool:
    low = error_message.lower()
    return any(marker in low for marker in (
        'insufficient_quota',
        'exceeded your current quota',
        'check your plan and billing',
    ))


def _mark_unavailable(reason: str) -> None:
    global _unavailable_reason
    with _unavailable_lock:
        if _unavailable_reason is None:
            _unavailable_reason = reason
            print('  [LLM FALLBACK] API quota exhausted (429); using deterministic crawler fallback.')


def _raise_if_unavailable() -> None:
    if _unavailable_reason is not None:
        raise LLMUnavailableError(_QUOTA_FALLBACK_MESSAGE)


def llm_is_unavailable() -> bool:
    return _unavailable_reason is not None


def _rate_limit_wait_seconds(error_message: str, attempt: int) -> float:
    """On a 429: prefer the wait the API suggests ('try again in 7.095s'), otherwise exponential backoff."""
    m = re.search(r"try again in ([\d.]+)s", error_message)
    if m:
        return float(m.group(1)) + 1.0
    return min(2.0 ** attempt * 2, 30.0)


def _sanitize_ssl_env() -> None:
    """Same as client.py: ignore SSL environment variables that point to a path that does not exist."""
    cert_file = os.getenv("SSL_CERT_FILE")
    if cert_file and not Path(cert_file).exists():
        os.environ.pop("SSL_CERT_FILE", None)
    cert_dir = os.getenv("SSL_CERT_DIR")
    if cert_dir and not Path(cert_dir).exists():
        os.environ.pop("SSL_CERT_DIR", None)


def get_openai_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _sanitize_ssl_env()
            spec = _PROVIDERS[PROVIDER]
            kwargs = {"api_key": os.getenv(spec["key_env"])}
            if spec["base_url"]:
                kwargs["base_url"] = spec["base_url"]
            _client = OpenAI(
                **kwargs,
                # This wrapper owns retries; avoid hidden SDK retries on hard quota errors.
                max_retries=int(os.getenv('LLM_SDK_MAX_RETRIES', '0')),
                timeout=float(os.getenv('LLM_REQUEST_TIMEOUT_SECONDS', '45')),
            )
            print(f"[LLM] provider={PROVIDER} model={DEFAULT_MODEL}")
    return _client


def _extract_json(text: str):
    """Tolerant parsing: try json.loads directly, then fish a ```json fence or the first {...}/[...] out of the text."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"The LLM reply could not be parsed as JSON: {text[:200]}")


def call_llm_json(prompt: str, model_name: str = DEFAULT_MODEL, system: str | None = None):
    """JSON-mode call (response_format=json_object); retries on failure and parses tolerantly."""
    _raise_if_unavailable()
    client = get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err = None
    for attempt in range(MAX_LLM_RETRY + 1):
        try:
            with _LLM_SEMAPHORE:
                # A worker may trip the breaker while this one waits for the semaphore.
                _raise_if_unavailable()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            return _extract_json(response.choices[0].message.content or "")
        except Exception as e:
            last_err = e
            msg = str(e)
            if isinstance(e, LLMUnavailableError):
                raise
            if _is_permanent_quota_error(msg):
                _mark_unavailable(msg)
                raise LLMUnavailableError(_QUOTA_FALLBACK_MESSAGE) from None
            if "rate_limit" in msg or "429" in msg:
                wait = _rate_limit_wait_seconds(msg, attempt)
                print(f"  [WAIT] LLM rate limited; retrying after {wait:.1f}s ({attempt + 1}/{MAX_LLM_RETRY + 1})")
                time.sleep(wait)
            else:
                print(f"  [WARN] LLM JSON call failed ({attempt + 1}/{MAX_LLM_RETRY + 1}): {msg[:200]}")
    raise last_err
