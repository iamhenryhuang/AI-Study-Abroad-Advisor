import json

import psycopg

from .connection import CONNECT_TIMEOUT, DATABASE_URL, PROJECT_ROOT, get_connection


# ── setup_db ─────────────────────────────────────────────────

def setup_db():
    """Check the connection and create the target database when needed."""
    from urllib.parse import urlparse, urlunparse
    url = DATABASE_URL
    if not url:
        print("Error: DATABASE_URL is not set in .env.")
        return False

    parsed = urlparse(url)
    db_name = parsed.path.lstrip("/")
    if not db_name:
        print("Error: no database name found in DATABASE_URL.")
        return False

    postgres_url = urlunparse(parsed._replace(path="/postgres"))
    try:
        print(f"Connecting to: {postgres_url.split('@')[-1]}")
        conn = psycopg.connect(postgres_url, connect_timeout=CONNECT_TIMEOUT)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if not cur.fetchone():
                print(f"Creating database {db_name} ...")
                cur.execute(f"CREATE DATABASE \"{db_name}\"")
                print(f"Created {db_name}.")
            else:
                print(f"Database {db_name} already exists.")
        conn.close()
        print("Connection test passed.")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


# ── init_schema ──────────────────────────────────────────────

def init_schema():
    """Reset all tables from db/init_db.sql (universities / programs and their child tables, web_pages / document_chunks / review_queue)."""
    conn = get_connection()
    if not conn:
        print("Set DATABASE_URL in .env.")
        return False
    try:
        # init_db.sql only does the DROP; the table definitions live in the shared schema_programs.sql
        # (data_crawler/db.py uses the same file, so the definition is not written twice and cannot drift).
        sql_paths = [
            PROJECT_ROOT / "db" / "init_db.sql",
            PROJECT_ROOT / "db" / "schema_programs.sql",
        ]
        for sql_path in sql_paths:
            if not sql_path.is_file():
                print(f"Not found: {sql_path}")
                return False
        # Run each file in one go (same approach as load_applicant_reports applying its migration):
        # psycopg3 accepts multi-statement strings and does not split $$...$$ function bodies.
        with conn.cursor() as cur:
            for sql_path in sql_paths:
                cur.execute(sql_path.read_text(encoding="utf-8"))
        conn.commit()
        print("Tables created or reset from init_db.sql + schema_programs.sql.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Failed to create tables: {e}")
        return False
    finally:
        conn.close()


def init_experience_schema():
    """Create the user experience table idempotently, leaving existing data alone."""
    try:
        from .experiences import ensure_experience_schema
        ensure_experience_schema()
        print("Created or confirmed the user_experiences table.")
        return True
    except Exception as e:
        print(f"Failed to initialize the experience table: {e}")
        return False


def clear_crawler_data():
    """Remove all crawler DB data, checkpoints and generated JSON; keep the schema and user experiences."""
    conn = get_connection()
    if not conn:
        print("Error: cannot connect to the database; no data or files were deleted.")
        return False

    try:
        with conn.cursor() as cur:
            # universities CASCADE clears programs, child tables, web_pages, chunks and review.
            cur.execute("TRUNCATE TABLE universities RESTART IDENTITY CASCADE")
            checkpoint_tables = []
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                if cur.fetchone()[0] is not None:
                    checkpoint_tables.append(table)
            if checkpoint_tables:
                # Table names come from the fixed whitelist above and never from outside input.
                cur.execute(f"TRUNCATE TABLE {', '.join(checkpoint_tables)}")
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        print(f"Failed to clear the database; JSON files were not deleted: {e}")
        return False
    conn.close()

    crawler_dir = PROJECT_ROOT / "data_crawler"
    patterns = (
        crawler_dir / "output" / "*.json",
        crawler_dir / "url" / "all_url" / "*.json",
        crawler_dir / "url" / "keep" / "*.json",
        crawler_dir / "url" / "drop" / "*.json",
    )
    removed = []
    try:
        for pattern in patterns:
            for path in pattern.parent.glob(pattern.name):
                path.unlink()
                removed.append(path.relative_to(PROJECT_ROOT).as_posix())
    except Exception as e:
        print(f"The database was cleared, but some JSON files could not be deleted: {e}")
        return False

    print("Cleared all crawler data and LangGraph checkpoints.")
    print(f"Deleted {len(removed)} crawler JSON files.")
    return True


def load_professors():
    """Load the faculty list from db/data/professors.json.

    The universities rows must already exist (created by data_crawler);
    this step does not create new schools."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from db.load_professors import load
    return load()


# ── verify ───────────────────────────────────────────────────

def verify():
    """Print the crawler DB's global structured data and an audit summary (without the full raw_text/chunks)."""
    if not DATABASE_URL:
        print("Error: DATABASE_URL is not set (.env).")
        return False
    try:
        conn = get_connection()
        if not conn:
            return False
        with conn.cursor() as cur:
            def show(label: str, sql: str):
                cur.execute(sql)
                columns = [desc.name for desc in cur.description]
                rows = [dict(zip(columns, row)) for row in cur.fetchall()]
                print(f"\n{'=' * 20} {label} ({len(rows)}) {'=' * 20}")
                print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))

            show("UNIVERSITIES", """
                SELECT id, school_id, name, domain, created_at
                FROM universities ORDER BY school_id
            """)
            show("PROGRAMS - ALL FIELDS", """
                SELECT u.school_id, p.*
                FROM programs p
                JOIN universities u ON u.id = p.university_id
                ORDER BY u.school_id, p.program_code
            """)
            show("GLOBAL EXTRACTIONS - INCLUDING SCHOOLS WITHOUT PROGRAMS", """
                SELECT u.school_id,
                       CASE WHEN EXISTS (
                           SELECT 1 FROM programs p WHERE p.university_id = ge.university_id
                       ) THEN true ELSE false END AS has_program,
                       ge.id, ge.scope, ge.program_code, ge.source_url,
                       ge.confidence, ge.extraction, ge.created_at, ge.updated_at
                FROM global_extractions ge
                JOIN universities u ON u.id = ge.university_id
                ORDER BY u.school_id, ge.scope, ge.id
            """)
            show("PROGRAM DEADLINES", """
                SELECT u.school_id, p.program_code, d.*
                FROM program_deadlines d
                JOIN programs p ON p.id = d.program_id
                JOIN universities u ON u.id = p.university_id
                ORDER BY u.school_id, p.program_code, d.semester, d.deadline_type
            """)
            show("PROGRAM SCHOLARSHIPS", """
                SELECT u.school_id, p.program_code, s.*
                FROM program_scholarships s
                JOIN programs p ON p.id = s.program_id
                JOIN universities u ON u.id = p.university_id
                ORDER BY u.school_id, p.program_code, s.name
            """)
            show("PROGRAM APPLICATION MATERIALS", """
                SELECT u.school_id, p.program_code, m.*
                FROM program_app_materials m
                JOIN programs p ON p.id = m.program_id
                JOIN universities u ON u.id = p.university_id
                ORDER BY u.school_id, p.program_code, m.material_type
            """)
            show("PROGRAM EVIDENCE", """
                SELECT u.school_id, p.program_code, e.*
                FROM program_evidence e
                JOIN programs p ON p.id = e.program_id
                JOIN universities u ON u.id = p.university_id
                ORDER BY u.school_id, p.program_code, e.category, e.field_name
            """)
            show("WEB PAGES - AUDIT", """
                SELECT u.school_id, w.id, w.program_id, w.url, w.passed_types,
                       w.char_count, left(w.raw_text, 300) AS raw_text_preview,
                       w.created_at
                FROM web_pages w
                LEFT JOIN universities u ON u.id = w.university_id
                ORDER BY u.school_id, w.id
            """)
            show("DOCUMENT CHUNKS - SUMMARY", """
                SELECT school_id, count(*) AS chunk_count,
                       count(*) FILTER (WHERE embedding IS NOT NULL) AS embedded_count,
                       min(created_at) AS first_created_at,
                       max(created_at) AS last_created_at
                FROM document_chunks GROUP BY school_id ORDER BY school_id
            """)
            show("REVIEW QUEUE", """
                SELECT u.school_id, rq.id, rq.page_id, rq.program_code,
                       rq.field_name, rq.field_value, rq.reason,
                       rq.source_excerpt, rq.status, rq.created_at
                FROM review_queue rq
                LEFT JOIN universities u ON u.id = rq.university_id
                ORDER BY u.school_id, rq.id
            """)

        conn.close()
        print("\nVerification passed: the data is in the database.")
        return True
    except psycopg.ProgrammingError as e:
        print(f"A table is missing or has the wrong structure: {e}")
        print("Run this first: python backend/scripts/run.py init-schema")
        return False
    except Exception as e:
        print(f"Connection or query error: {e}")
        return False
