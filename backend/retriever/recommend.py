"""Upload scores -> school recommendations: tier schools into reach/match/safety by the 1point3acres admission medians,
and pull real admission cases with similar scores from applicant_reports as evidence.

The tiering logic is a pure function (the distribution can be injected); the DB case query is a separate fetch_nearby_cases.
"""
from __future__ import annotations

import json

from database.connection import get_connection
from settings import PROJECT_ROOT

_DIST_PATH = PROJECT_ROOT / "db" / "data" / "1point3_distribution.json"

# Representative ETS IELTS->TOEFL concordance values
_IELTS_TOEFL = {9.0: 119, 8.5: 116, 8.0: 112, 7.5: 105, 7.0: 98, 6.5: 86, 6.0: 69}


def _ielts_to_toefl(ielts) -> int | None:
    if ielts is None:
        return None
    try:
        key = round(float(ielts) * 2) / 2      # snap to 0.5 steps
    except (TypeError, ValueError):
        return None
    return _IELTS_TOEFL.get(key)


def _normalize_profile(profile: dict) -> dict:
    """Return {gpa, toefl, gre}; when toefl is missing but ielts is given, convert it and fill it in."""
    gpa   = profile.get("gpa")
    toefl = profile.get("toefl")
    gre   = profile.get("gre")
    if toefl is None and profile.get("ielts") is not None:
        toefl = _ielts_to_toefl(profile.get("ielts"))
    return {"gpa": gpa, "toefl": toefl, "gre": gre}


def classify_tier(profile: dict, medians: dict) -> str | None:
    """Tier by the share of dimensions met: mostly met -> safety, about half -> match, mostly missed -> reach; no comparable dimension -> None."""
    dims = []
    for key in ("gpa", "toefl", "gre"):
        u, m = profile.get(key), medians.get(f"median_{key}")
        if u is not None and m is not None:
            dims.append((u, m))
    if not dims:
        return None
    meets = sum(1 for u, m in dims if u >= m)
    ratio = meets / len(dims)
    if ratio >= 0.7:
        return "保底"
    if ratio >= 0.35:
        return "適中"
    return "衝刺"


def _load_distribution() -> dict:
    """Read the distribution json -> {school_id: {school_id, name, median_gpa, median_toefl, median_gre}}."""
    try:
        raw = json.loads(_DIST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Recommend] Failed to read the distribution: {e}")
        return {}
    out = {}
    for name, info in raw.get("university_admissions_data", {}).items():
        sid = info.get("school_id")
        if not sid:
            continue
        out[sid] = {
            "school_id":     sid,
            "name":          name,
            "median_gpa":    info.get("median_gpa"),
            "median_toefl":  info.get("median_toefl"),
            "median_gre":    info.get("median_gre"),
        }
    return out


def _comparison(profile: dict, med: dict) -> list[str]:
    """Build the list of "your score vs the median" comparison strings."""
    parts = []
    for key, label in (("gpa", "GPA"), ("toefl", "TOEFL"), ("gre", "GRE")):
        u, m = profile.get(key), med.get(f"median_{key}")
        if u is not None and m is not None:
            parts.append(f"{label} 你 {u} / 中位 {m}")
    return parts


def recommend(profile: dict, distribution: dict | None = None, per_tier: int = 3) -> dict:
    """Tier the schools, taking at most per_tier schools in each tier. The distribution can be injected for tests."""
    norm = _normalize_profile(profile)
    dist = distribution if distribution is not None else _load_distribution()
    tiers: dict[str, list] = {"衝刺": [], "適中": [], "保底": []}
    for sid, med in dist.items():
        tier = classify_tier(norm, med)
        if tier is None:
            continue
        tiers[tier].append({
            "school_id":  sid,
            "name":       med.get("name", sid.upper()),
            "medians":    med,
            "comparison": _comparison(norm, med),
        })
    for tier in tiers:
        tiers[tier] = tiers[tier][:per_tier]
    return tiers


def fetch_nearby_cases(school_id: str, gpa, limit: int = 3) -> list[dict]:
    """Find admitted cases at that school in applicant_reports with a similar GPA (+/-0.2). Returns [] on failure."""
    conn = get_connection()
    if not conn:
        return []
    try:
        conn.read_only = True
        with conn.cursor() as cur:
            if gpa is not None:
                cur.execute(
                    "SELECT gpa, decision, notes FROM applicant_reports "
                    "WHERE school_id=%s AND decision IN ('accepted','offer','ad_no_fund','ad_small_fund') "
                    "AND gpa IS NOT NULL AND ABS(gpa - %s) <= 0.2 "
                    "ORDER BY ABS(gpa - %s) LIMIT %s",
                    (school_id, gpa, gpa, limit),
                )
            else:
                cur.execute(
                    "SELECT gpa, decision, notes FROM applicant_reports "
                    "WHERE school_id=%s AND decision IN ('accepted','offer','ad_no_fund','ad_small_fund') "
                    "ORDER BY id DESC LIMIT %s",
                    (school_id, limit),
                )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    except Exception as e:
        print(f"[Recommend] Case query failed ({school_id}): {e}")
        return []
    finally:
        conn.close()

    return rows
