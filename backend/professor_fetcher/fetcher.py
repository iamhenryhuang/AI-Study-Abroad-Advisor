"""
Search strategy:
  1. search_professor_id()                    -- search the google_scholar engine for "{name}" "{school}" and
                                                 take the author_id from publication_info.authors[].link in the paper results
  2. fetch_author_profile_and_recent_papers() -- open the profile page by author_id for the profile and the last two years of papers
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime

import requests

import settings  # noqa: F401  (loads backend/.env)

SERPAPI_KEY: str = os.environ.get("SERPAPI_KEY", "").strip()
SERPAPI_BASE = "https://serpapi.com/search"

_THIS_YEAR = datetime.now().year
RECENT_YEARS_CUTOFF = _THIS_YEAR - 2   # e.g. 2026 -> keep 2024, 2025, 2026

_PLACEHOLDER_KEYS = {"your_serpapi_key_here", "your_key_here", "", "YOUR_KEY"}


# -- Helpers ------------------------------------------------------------------

def _validate_key() -> None:
    """Check that SERPAPI_KEY is valid."""
    if not SERPAPI_KEY or SERPAPI_KEY in _PLACEHOLDER_KEYS or len(SERPAPI_KEY) < 20:
        raise RuntimeError(
            "\nSERPAPI_KEY is not set or is still the placeholder value!\n\n"
            "Set it up as follows:\n"
            "  1. Get an API key at https://serpapi.com\n"
            "  2. Set it in backend/.env: SERPAPI_KEY=your-actual-key\n"
        )


def _get(params: dict) -> dict:
    """Call SerpAPI with retries and error handling, and return the parsed JSON."""
    _validate_key()
    all_params = {**params, "api_key": SERPAPI_KEY}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Raise the timeout to 120 s to cope with a slow SerpAPI server
            resp = requests.get(SERPAPI_BASE, params=all_params, timeout=120)

            if resp.status_code == 400:
                try:
                    data = resp.json()
                    err = data.get("error", resp.text[:200])
                except Exception:
                    err = resp.text[:200]
                raise requests.HTTPError(
                    f"SerpAPI 400 Bad Request (engine={params.get('engine')}): {err}",
                    response=resp,
                )

            resp.raise_for_status()
            return resp.json()

        except requests.HTTPError as e:
            # 400, 401 and 403 are client errors and usually should not be retried
            if e.response is not None and e.response.status_code in (400, 401, 403):
                raise
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt + 1}/{max_retries}] HTTPError: {str(e)[:100]}")
            time.sleep(2 ** attempt)

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt + 1}/{max_retries}] Network error: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(min(2 ** (attempt + 1), 10))

    return {}


def _extract_author_id_from_url(url: str) -> str | None:
    """Extract the user= parameter (author_id) from a Google Scholar URL."""
    if not url:
        return None
    m = re.search(r"[?&]user=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


# -- Main feature: look up the author_id -----------------------------------------

def _normalize_name(s: str) -> str:
    """Strip hyphens and periods and lowercase, for fuzzy comparison. Example: 'F.-F. Li' -> 'ff li'"""
    return re.sub(r"[-.]", "", s.lower())


def _name_score(search_name: str, candidate_name: str) -> float:
    """
    Returns a 0-1 score for how similar candidate_name and search_name are.
    It considers both exact word matches and fuzzy matches after normalization.
    """
    s_parts  = search_name.lower().split()          # ["fei-fei", "li"]
    c_parts  = candidate_name.lower().split()       # ["fei-fei", "li"] or ["fei", "li"]
    c_norm   = _normalize_name(candidate_name)      # "feifei li" or "fei li"

    matched = 0.0
    for sp in s_parts:
        if sp in c_parts:
            matched += 1.0                          # exact word match
        elif _normalize_name(sp) in c_norm:
            matched += 0.8                          # match after normalization (hyphens/periods removed)
        elif sp[0] in [p[0] for p in c_parts]:
            matched += 0.3                          # initial match (abbreviated names)

    return matched / len(s_parts) if s_parts else 0.0


def _search_via_papers(name: str, affiliation: str) -> str | None:
    """
    Search the paper results and take the author_id from the author links.
    Uses a stricter name-matching threshold and prints a warning when several candidates tie.
    """
    # No quotes, so Google Scholar does a fuzzy search; _name_score filters for the right author
    query = f'{name} {affiliation}'.strip() if affiliation else name
    print(f"  [Papers] Searching: {query!r}")

    try:
        data = _get({
            "engine": "google_scholar",
            "q": query,
            "hl": "en",
            "num": 10,
        })
    except Exception as e:
        print(f"  [Papers] Search failed: {e}")
        return None

    organic = data.get("organic_results", [])
    if not organic:
        print("  [Papers] Search returned nothing")
        return None

    candidates: list[dict] = []
    for result in organic:
        authors = result.get("publication_info", {}).get("authors", [])
        for author in authors:
            a_name = author.get("name", "")
            link   = author.get("link", "") or author.get("serpapi_scholar_link", "")

            aid = _extract_author_id_from_url(link)
            if not aid:
                m = re.search(r"author_id=([A-Za-z0-9_-]+)", link)
                aid = m.group(1) if m else None
            if not aid:
                continue

            score = _name_score(name, a_name)
            # Require high similarity (>= 0.6) to be a candidate, to cover abbreviated names such as "A Ng" and "AY Ng"
            if score < 0.6:
                continue

            if not any(c["id"] == aid for c in candidates):
                candidates.append({"name": a_name, "id": aid, "score": score})

    if not candidates:
        print(f"  [Papers] Could not find an author_id for {name!r}")
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    if len(candidates) > 1 and abs(candidates[1]["score"] - best["score"]) < 0.1:
        print("  [Papers] Several candidates have similar scores; the wrong person may have been picked:")
        for c in candidates[:3]:
            print(f"      - {c['name']}  id={c['id']}  score={c['score']:.2f}")

    print(f"  [Papers] Picked id={best['id']} ({best['name']}  score={best['score']:.2f})")
    return best["id"]


def search_professor_id(name: str, affiliation: str = "") -> str | None:
    """
    Look up a professor's Google Scholar author_id.

    Note: the google_scholar_profiles engine has been retired, so this searches papers directly.

    Args:
        name:        the professor's full name
        affiliation: school name or keyword
    """
    return _search_via_papers(name, affiliation)


def _extract_recent_papers(
    data: dict,
    professor_name: str = "",
    cutoff_year: int | None = None,
    max_papers: int = 20,
) -> list[dict]:
    cutoff = cutoff_year if cutoff_year is not None else RECENT_YEARS_CUTOFF

    papers = []
    for article in data.get("articles", []):
        year = None
        year_str = str(article.get("year", ""))
        m = re.search(r"\b(20\d{2}|19\d{2})\b", year_str)
        if m:
            year = int(m.group(1))

        if year and year < cutoff:
            continue

        cited_val = 0
        cited_raw = article.get("cited_by", {})
        if isinstance(cited_raw, dict):
            cited_val = cited_raw.get("value", 0) or 0

        papers.append({
            "title":          article.get("title", ""),
            "link":           article.get("link", ""),
            "authors":        article.get("authors", professor_name),
            "publication":    article.get("publication", ""),
            "year":           year,   # None when Scholar lists no year; never filled in with a guess
            "cited_by_value": cited_val,
            "snippet":        article.get("description", ""),
        })
        if len(papers) >= max_papers:
            break

    print(f"  [Papers] collected {len(papers)} papers since {cutoff}")
    return papers


def fetch_author_profile_and_recent_papers(
    author_id: str,
    professor_name: str = "",
    cutoff_year: int | None = None,
    max_papers: int = 20,
) -> tuple[dict, list[dict]]:
    """Fetch author metadata and recent papers with one SerpAPI author request."""
    try:
        data = _get({
            "engine":    "google_scholar_author",
            "author_id": author_id,
            "hl":        "en",
            "sort":      "pubdate",
        })
    except Exception as e:
        print(f"  author fetch failed: {e}")
        return {"search_parameters": {"author_id": author_id}}, []

    papers = _extract_recent_papers(
        data=data,
        professor_name=professor_name,
        cutoff_year=cutoff_year,
        max_papers=max_papers,
    )
    return data, papers
