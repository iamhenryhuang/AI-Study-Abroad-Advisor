"""Public runtime API for executing the retriever agent."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

# Load backend/.env explicitly: a bare load_dotenv() only looks for a .env in the CWD (the project root has none),
# and it used to work only as a side effect of other modules being imported first.
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_BACKEND_ROOT / ".env")

from .graph import _get_graph
from .state import (
    AgentCancelledError,
    clear_execution_context,
    create_initial_state,
    set_execution_context,
)


def run_agent(
    query: str,
    max_steps: int = 10,
    verbose: bool = True,
    on_event: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str | None:
    """Execute the Agentic RAG workflow and return its final answer."""
    set_execution_context(on_event, cancel_event)

    try:
        if verbose:
            print(f"\n{'=' * 60}")
            print("   Agentic RAG started")
            print(f"   Agentic RAG start time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Question: {query}")
            print(f"{'=' * 60}")

        result_state = _get_graph().invoke(
            create_initial_state(query),
            {"recursion_limit": max_steps * 4},
        )

        if verbose:
            total_docs = len(result_state.get("verified_docs", []))
            print(f"\n{'=' * 60}")
            print(f"Done! Combined {total_docs} de-duplicated items")
            print(f"{'=' * 60}")
            print(f"   Agentic RAG end time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return result_state.get("final_answer") or None

    except AgentCancelledError:
        print("[Agent] Received a cancel signal; ending early.")
        return None

    finally:
        clear_execution_context()
