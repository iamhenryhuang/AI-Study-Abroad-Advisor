"""Command-line entry point: database setup and single-question runs. See COMMANDS below."""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for path in (BACKEND_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from database.operations import (clear_crawler_data, init_experience_schema, init_schema,
                                 load_professors, setup_db, verify)
from retriever.rag_pipeline import run_agent_pipeline, run_rag_pipeline

# The three schools the live demo is fixed to (root URLs are defined in data_crawler/config/root_url.py)
DEMO_SCHOOLS = ("purdue", "gatech", "stanford")


def init_full() -> bool:
    """One-step DB skeleton and community data: connect -> rebuild the programs family (empty tables) ->
    user experience table -> community application reports (including the migration). Stops at the first failing step.

    School/program data is not written in this step: it is written after data_crawler really crawls purdue / gatech /
    stanford (see the hint below), so this only runs init-schema to create empty tables.
    Faculty lists attach to existing universities, so load-professors runs separately after the crawler."""
    from db.loaders.load_applicant_reports import load as load_applicant_reports

    steps = [
        ("setup",              setup_db),
        ("init-schema",        init_schema),    # rebuild the programs family (empty tables, waiting for the crawler)
        ("init-experience",    init_experience_schema),
        ("load-applicant",     lambda: load_applicant_reports(create_table=True)),
    ]
    for name, fn in steps:
        print(f"\n-- init-full: {name} --")
        if fn() is False:
            print(f"init-full stopped at {name}: failed.")
            return False
    print("\ninit-full done: the empty programs family / user_experiences / applicant_reports are ready.")
    print("\nNext, crawl the school data (once per school, with embedding on):")
    for sid in DEMO_SCHOOLS:
        print(f"  python -m data_crawler.main --school-id {sid} --enable-embedding")
    print("After crawling, load the faculty list:")
    print("  python backend/run.py load-professors")
    return True


COMMANDS = {
    "setup":     ("Check the connection and create the database",                       setup_db),
    "init-schema": ("Drop and recreate the crawler tables (deletes crawled data)",      init_schema),
    "init-experience": ("Create the user experience table (idempotent, keeps data)", init_experience_schema),
    "load-professors": ("Load the faculty list from db/data/professors.json (universities must exist first)", load_professors),
    "verify-db": ("Check that the SQL data has been written",                     verify),
    "clear-crawler-data": ("Clear all crawler DB data/checkpoints/result JSON (requires --yes)", None),
    "search":    ("Run a text-to-SQL query test [query]",           None),  # special handling
    "rag":       ("One SQL retrieval + answer generation [query]",            None),  # special handling
    "agent":     ("Agentic RAG LangGraph Loop [query] [--max-steps N]", None),  # special handling
    "init-full": ("One-step DB skeleton: empty programs family + experience table + community reports (school data comes from the crawler)", init_full),
}


def main():
    # Windows may default redirected console output to cp950, while crawled
    # university pages legitimately contain symbols such as ≥ and em dashes.
    # Keep CLI output deterministic and avoid crashing after successful retrieval.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Available commands:")
        for cmd, (desc, _) in COMMANDS.items():
            print(f"  {cmd:<19}  {desc}")
        sys.exit(1)

    cmd = sys.argv[1]
    _, runner = COMMANDS[cmd]

    if cmd == "clear-crawler-data":
        if "--yes" not in sys.argv[2:]:
            print("Refusing to run: this would permanently delete all crawler data and result files.")
            print("To confirm, run: python backend/run.py clear-crawler-data --yes")
            sys.exit(2)
        ok = clear_crawler_data()
    elif cmd in ["search", "rag", "agent"]:
        max_steps = 5
        if "--max-steps" in sys.argv:
            idx = sys.argv.index("--max-steps")
            if idx + 1 < len(sys.argv):
                try:
                    max_steps = int(sys.argv[idx + 1])
                except ValueError:
                    pass

        raw_args = sys.argv[2:]
        args_clean = []
        i = 0
        while i < len(raw_args):
            token = raw_args[i]
            if token == "--max-steps":
                i += 2
                continue
            args_clean.append(token)
            i += 1

        query = " ".join(args_clean).strip()

        if not query:
            query = input("Enter a query question: ").strip()
            if not query:
                print("No question entered; stopping.")
                sys.exit(0)

        if cmd == "search":
            from retriever.search.sql_search import sql_search
            results, sql = sql_search(query)
            print(f"SQL: {sql}")
            for row in results:
                print(row)
            ok = bool(results)
        elif cmd == "agent":
            ok = run_agent_pipeline(query, max_steps=max_steps)
        else:
            ok = run_rag_pipeline(query)
    else:
        ok = runner()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
