"""CLI entry point.

Examples (run a small crawl first, then widen it):
    python -m data_crawler.main --school-id purdue --max-depth 1 --max-pages 10 --dry-run
    python -m data_crawler.main --school-id ucla --max-depth 2
    python -m data_crawler.main --school-id ucla --resume            # resume from the checkpoint
    python -m data_crawler.main --school-id ucla --enable-embedding  # turn embedding on

checkpointer:
    DATABASE_URL set and not --dry-run -> PostgresSaver (thread_id = school_id)
    --dry-run or no DATABASE_URL       -> MemorySaver (cannot resume across processes)
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "backend" / ".env")

RECURSION_LIMIT = 300


def parse_args():
    ap = argparse.ArgumentParser(description="AI Study Abroad Consultant -- LangGraph data extraction pipeline")
    ap.add_argument("--school-id", required=True, help="a school_id from config/root_url.py (one school)")
    ap.add_argument("--root-url", action="append", default=None,
                    help="crawl only the given root, overriding root_url.py; can be repeated")
    ap.add_argument("--max-depth", type=int, default=None, help="BFS depth limit (default: CONFIG.MAX_DEPTH)")
    ap.add_argument("--max-pages", type=int, default=None, help="total page limit (default: CONFIG.MAX_PAGES)")
    ap.add_argument("--skip-recent-hours", type=int, default=0,
                    help="when > 0 enables SKIP_RECENT: skip the whole school if last_extracted_at in the DB is within N hours")
    ap.add_argument("--enable-embedding", action="store_true", help="turn on Node 12 embedding (off by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not write the DB; write the result to data_crawler/output/{school_id}_result.json")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint (PostgresSaver only)")
    ap.add_argument("--thread-id", default=None, help="override thread_id (default = school_id)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Import late so --help does not wait for playwright/langgraph to load
    from .graph import build_school_graph

    initial_state = {
        "school_id": args.school_id,
        "root_urls_override": args.root_url,
        "max_depth": args.max_depth,
        "max_pages": args.max_pages,
        "skip_recent_hours": args.skip_recent_hours,
        "enable_embedding": args.enable_embedding
                            or os.getenv("ENABLE_EMBEDDING", "").lower() in ("1", "true", "on"),
        "dry_run": args.dry_run,
    }

    thread_id = args.thread_id or args.school_id
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    graph_input = None if args.resume else initial_state

    database_url = os.getenv("DATABASE_URL")
    use_postgres = bool(database_url) and not args.dry_run

    if use_postgres:
        from datetime import datetime
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(database_url) as checkpointer:
            checkpointer.setup()
            app = build_school_graph(checkpointer)
            # A fresh non-resume run: if the same thread already has old records, open a new thread,
            # so accumulating state (page_results etc.) does not pile up across runs
            if not args.resume and checkpointer.get(config) is not None:
                thread_id = f"{args.school_id}-{datetime.now():%Y%m%d%H%M%S}"
                print(f"[INFO] Thread '{args.school_id}' already has earlier run records; using thread_id={thread_id} this time"
                      f" (add --resume to continue the old run)")
                config = {"configurable": {"thread_id": thread_id},
                          "recursion_limit": RECURSION_LIMIT}
            final_state = app.invoke(graph_input, config)
    else:
        if args.resume:
            print("[WARN] --resume needs PostgresSaver (DATABASE_URL set and not --dry-run); running from scratch instead")
        from langgraph.checkpoint.memory import MemorySaver
        app = build_school_graph(MemorySaver())
        final_state = app.invoke(initial_state, config)

    summary = final_state.get("summary", {})
    print(f"\n[OK] Done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
