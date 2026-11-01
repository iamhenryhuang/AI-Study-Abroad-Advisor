"""
Write professors (the faculty seed data) from db/data/professors.json.

Every row must map to an existing universities.school_id (created by data_crawler first); this script does not create new schools.

Run: python db/load_professors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_SCRIPTS = _PROJECT_ROOT / "backend" / "scripts"
if str(_BACKEND_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SCRIPTS))

from database.connection import get_connection

DATA_PATH = Path(__file__).resolve().parent / "data" / "professors.json"


def load() -> bool:
    if not DATA_PATH.exists():
        print(f"Not found: {DATA_PATH}. See db/data/README.md for the format.")
        return False
    professors = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    conn = get_connection()
    if not conn:
        print("Set DATABASE_URL in backend/.env.")
        return False

    try:
        with conn.cursor() as cur:
            skipped = []
            written = 0
            for p in professors:
                cur.execute(
                    "SELECT id FROM universities WHERE school_id = %s",
                    (p["school_id"],),
                )
                row = cur.fetchone()
                if not row:
                    skipped.append(f"{p['name']} ({p['school_id']}: the school does not exist)")
                    continue
                university_id = row[0]

                cur.execute(
                    """
                    INSERT INTO professors (university_id, name, title, research_areas, profile_url)
                    VALUES (%(university_id)s, %(name)s, %(title)s, %(research_areas)s, %(profile_url)s)
                    ON CONFLICT (university_id, name) DO UPDATE SET
                        title          = EXCLUDED.title,
                        research_areas = EXCLUDED.research_areas,
                        profile_url    = EXCLUDED.profile_url
                    """,
                    {
                        "university_id": university_id,
                        "name": p["name"],
                        "title": p.get("title") or None,
                        "research_areas": p.get("research_areas", []),
                        "profile_url": p.get("profile_url") or None,
                    },
                )
                written += 1

        conn.commit()
        print(f"Wrote {written} professors.")
        if skipped:
            print(f"Skipped {len(skipped)} rows (the school is not in universities):")
            for s in skipped:
                print(f"  - {s}")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Write failed: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(0 if load() else 1)
