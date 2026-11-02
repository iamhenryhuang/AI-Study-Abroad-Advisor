"""Clean and load community application-report data into the applicant_reports table.

Sources (all experience-type, unofficial data):
  db/data/gradcafe_results.json -- GradCafe (English, structured)
  db/data/1point3.json          -- 1point3acres (Simplified Chinese, semi-structured)

Run:
  python db/loaders/load_applicant_reports.py                 # the table must already exist
  python db/loaders/load_applicant_reports.py --create-table  # also create the table first

See the _clean_* functions and the comments in db/schema/applicant_reports.sql for the cleaning strategy.
Safe to re-run: (source, source_url) is UNIQUE, and ON CONFLICT updates the existing row.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _PROJECT_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database.connection import get_connection
from retriever.school_aliases import detect_school_ids

_DATA_DIR = _PROJECT_ROOT / "db" / "data"
GRADCAFE_PATH = _DATA_DIR / "gradcafe_results.json"
ONEPOINT3_PATH = _DATA_DIR / "1point3.json"
TABLE_SQL_PATH = _PROJECT_ROOT / "db" / "schema" / "applicant_reports.sql"


# --- Shared cleaning helpers -------------------------------------------------------

def _text(value) -> str:
    """A user-built JSON field as stripped text; scores may be written as numbers rather than strings."""
    return "" if value is None else str(value).strip()


def _match_school_id(school_raw: str) -> str | None:
    """Map a raw school name to a school_id with the alias table; return None when nothing matches (keeping school_raw is enough)."""
    detected = detect_school_ids(school_raw)
    return detected[0] if detected else None


def _parse_gpa(raw: str | None) -> float | None:
    """Extract a "4.0-scale-looking" value (0-4.3) from a raw GPA string; return None when nothing can be extracted.

    Formats covered:
      '3.63'                → 3.63
      '3.98||3.83'          -> take the segment within 0-4.3 (there may be several; take the last reasonable value)
      '75||3.1||20/47'      -> 75 is a percentage scale and skipped, 20/47 is a rank and skipped, take 3.1
      '3.69/5||3.5'         -> '3.69/5' is a 5-point scale and skipped (contains /), take 3.5
      '9.05' / '6.74'       -> beyond 4.3, treated as not a 4.0 scale -> None
    """
    if not raw:
        return None
    candidate: float | None = None
    for seg in str(raw).split("||"):
        seg = seg.strip()
        # A segment containing '/' is a rank (20/47) or a non-4.0 scale (3.69/5); always skip it
        if "/" in seg:
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)$", seg)
        if not m:
            continue
        val = float(m.group(1))
        if 0 < val <= 4.3:          # within the reasonable 4.0-scale range
            candidate = val         # when several segments are reasonable, take the last (usually the more precise major GPA)
    return candidate


# --- gradcafe cleaning -------------------------------------------------------------

_GRADCAFE_DECISION = {
    "Accepted": "accepted",
    "Rejected": "rejected",
    "Wait listed": "waitlisted",
    "Interview": "interview",
    "Other": "other",
}
_GRADCAFE_LEVEL = {
    "Masters": "masters", "PhD": "phd", "MFA": "mfa",
    "PsyD": "psyd", "EdD": "edd", "Other": "other",
}


def _clean_gradcafe(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        school_raw = (r.get("school") or "").strip()
        if not school_raw:
            continue
        ugpa = r.get("ugpa")

        # Status (International/American, filled in 971/995 rows) and the three GRE sections (~70 rows)
        # are merged into notes -- they are very valuable for analyzing "admitted applicants' background" and must not be dropped during cleaning.
        extras = []
        status = (str(r.get("status") or "")).strip()
        if status:
            extras.append(f"身分:{status}")
        def _gre_part(prefix: str, val) -> str:
            s = str(val or "").strip()
            return f"{prefix}{s}" if s and s not in ("0", "0.0", "0.00") else ""

        gre = " ".join(p for p in (
            _gre_part("Q", r.get("greq")),
            _gre_part("V", r.get("grev")),
            _gre_part("W", r.get("grew")),
        ) if p)
        if gre:
            extras.append(f"GRE:{gre}")
        base_notes = (r.get("notes") or "").strip()
        notes = " | ".join(filter(None, [" ".join(extras) or None, base_notes])) or None

        out.append({
            "source":       "gradcafe",
            "source_url":   f"gradcafe:{r.get('id')}",  # no real URL, so a stable id serves as the dedup key
            "school_raw":   school_raw,
            "school_id":    _match_school_id(school_raw),
            "program":      (r.get("program") or "").strip() or None,
            "degree_level": _GRADCAFE_LEVEL.get(r.get("level"), None),
            "decision":     _GRADCAFE_DECISION.get(r.get("decision"), "other"),
            "gpa":          _parse_gpa(ugpa),
            "gpa_raw":      ugpa or None,
            "season":       (r.get("season") or "").strip() or None,
            "notes":        notes,
        })
    return out


# --- 1point3 cleaning --------------------------------------------------------------

_ONEPOINT3_RESULT = {
    "Offer":  "offer",
    "AD无奖": "ad_no_fund",
    "AD小奖": "ad_small_fund",
    "Reject": "rejected",
    "Waiting": "waitlisted",
}

# Forum-crawler noise: remove whole lines/fragments
_NOISE_PATTERNS = [
    re.compile(r"[.．・]*\s*(?:Waral|check 1point3acres|1point3acres|acres)\b.*", re.IGNORECASE),
    re.compile(r"\.\s*[Χχ█]+"),
    re.compile(r"\bWaral\s+d[иeи]+"),
]


def _clean_content(text: str | None) -> str | None:
    """Remove the forum-crawler noise from 1point3 content and compress whitespace. Empty content returns None."""
    if not text:
        return None
    for pat in _NOISE_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"[?？]{2,}", "", text)     # garbled '??'
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _clean_1point3(data: dict) -> list[dict]:
    out = []
    for school_key, posts in data.items():
        for p in posts:
            url = _text(p.get("url"))
            school_raw = _text(p.get("school") or school_key)
            if not school_raw:
                continue
            gpa_raw = p.get("gpa") or None
            # notes: title + cleaned short body (most post bodies are very short, so they are merged into one readable summary)
            title = _text(p.get("title"))
            content = _clean_content(p.get("content"))
            extras = []
            for label, field in (("專業", "major"), ("本科", "undergrad_school"), ("TOEFL", "toefl"), ("GRE", "gre")):
                if _text(p.get(field)):
                    extras.append(f"{label}:{_text(p.get(field))}")
            notes = " | ".join(filter(None, [title, " ".join(extras) or None, content])) or None

            out.append({
                "source":       "1point3",
                "source_url":   url or f"1point3:{school_key}:{title[:40]}",
                "school_raw":   school_raw,
                "school_id":    _match_school_id(school_raw),
                "program":      _text(p.get("major")) or None,
                "degree_level": None,   # the forum does not distinguish masters/phd, so leave it empty
                "decision":     _ONEPOINT3_RESULT.get(p.get("result"), "other"),
                "gpa":          _parse_gpa(gpa_raw),
                "gpa_raw":      gpa_raw,
                "season":       None,
                "notes":        notes,
            })
    return out


# --- Loading ----------------------------------------------------------------------

_INSERT_SQL = """
    INSERT INTO applicant_reports
        (source, source_url, school_raw, school_id, program, degree_level,
         decision, gpa, gpa_raw, season, notes)
    VALUES
        (%(source)s, %(source_url)s, %(school_raw)s, %(school_id)s, %(program)s,
         %(degree_level)s, %(decision)s, %(gpa)s, %(gpa_raw)s, %(season)s, %(notes)s)
    ON CONFLICT (source, source_url) DO UPDATE SET
        school_raw = EXCLUDED.school_raw, school_id = EXCLUDED.school_id,
        program = EXCLUDED.program, degree_level = EXCLUDED.degree_level,
        decision = EXCLUDED.decision, gpa = EXCLUDED.gpa, gpa_raw = EXCLUDED.gpa_raw,
        season = EXCLUDED.season, notes = EXCLUDED.notes
"""


def load(create_table: bool = False) -> bool:
    conn = get_connection()
    if not conn:
        print("Set DATABASE_URL in backend/.env.")
        return False

    try:
        with conn.cursor() as cur:
            if create_table:
                if not TABLE_SQL_PATH.is_file():
                    print(f"Not found: {TABLE_SQL_PATH}")
                    return False
                cur.execute(TABLE_SQL_PATH.read_text(encoding="utf-8"))
                conn.commit()   # keep the table even if there is no data to load yet
                print("Created or confirmed the applicant_reports table.")

            records: list[dict] = []
            if GRADCAFE_PATH.is_file():
                records += _clean_gradcafe(json.loads(GRADCAFE_PATH.read_text(encoding="utf-8")))
            if ONEPOINT3_PATH.is_file():
                records += _clean_1point3(json.loads(ONEPOINT3_PATH.read_text(encoding="utf-8")))

            if not records:
                print("No source data found (the json files under db/data/).")
                return False

            for rec in records:
                cur.execute(_INSERT_SQL, rec)

        conn.commit()
        matched = sum(1 for r in records if r["school_id"])
        with_gpa = sum(1 for r in records if r["gpa"] is not None)
        print(f"Loaded {len(records)} application reports"
              f" (matched to a known school_id: {matched}, GPA parsed: {with_gpa}).")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Load failed: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(0 if load(create_table="--create-table" in sys.argv) else 1)
