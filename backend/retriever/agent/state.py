"""Agent state and execution context."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Callable, Optional, TypedDict


class AgentCancelledError(Exception):
    """Raised on a cancel event, to stop the agent run quickly."""


# ContextVar (not threading.local): LangGraph runs parallel nodes in a ThreadPoolExecutor
# (such as search + experience_search at the same time) and copies the current context to the worker thread
# with contextvars.copy_context() before submitting; threading.local does not cross that executor,
# so the worker thread would read an empty value and _emit would silently drop thinking/tool_call events.
_on_event_var: ContextVar[Optional[Callable[[dict], None]]] = ContextVar("on_event", default=None)
_cancel_event_var: ContextVar[Optional[threading.Event]] = ContextVar("cancel_event", default=None)


def _check_cancel() -> None:
    """Raise AgentCancelledError immediately if the cancel event has been set."""
    cancel_event = _cancel_event_var.get()
    if cancel_event is not None and cancel_event.is_set():
        raise AgentCancelledError("The agent was cancelled")

# ─── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    original_query:    str
    sub_queries:       list[str]
    collected_docs:    list[dict]    # SQL results (the whole list is overwritten each round by searcher_node, not stacked on the previous round when retrying)
    fulltext_docs:     list[dict]    # full-text results (added by fulltext_search_node only when SQL is not enough)
    fulltext_done:     bool          # whether the full-text fallback was already run for this round's sub-questions (avoids doing it twice)
    extension_docs:    list[dict]    # professor data (overwritten each round by extension_function_node, not stacked on the previous round when retrying)
    experience_docs:   list[dict]    # application experience reports (overwritten by experience_search_node when needs_experience is set)
    final_answer:      str
    professor_query:   dict | None   # professor query detected by the decomposer {name, school, school_id}
    professor_list_query: dict | None # professor list query detected by the decomposer {school_id} (queries the professors table)
    needs_sql_search:  bool         # whether the decomposer decided to query the programs application-requirement data
    needs_experience:  bool         # whether the decomposer decided the question asks for experiences, so applicant_reports + user_experiences are queried
    mentioned_school_ids: list[str]   # school_ids the decomposer found in the known list (does not mean the database covers them)
    mentioned_school_names: list[str] # school names as written that the decomposer detected (not limited to the known list)
    verified_docs:     list[dict]    # data de-duplicated by the Verifier (search + extension merged)
    is_sufficient:     bool          # whether the Verifier judged the data enough to answer
    insufficiency_reason: str        # short reason when the data is insufficient or off-topic
    retry_count:       int           # number of retries so far (how many times the Refiner re-queried; at most one round)
    generated_answer:  bool          # whether the finalizer really called the LLM to generate an answer (decides whether the Critic runs)
    wants_recommendation: bool       # the user supplied scores and asked for school recommendations
    profile:           dict          # extracted scores {gpa, ielts, toefl, gre}
    recommend_docs:    list[dict]    # recommendation data produced by recommend_node


# --- Helpers ------------------------------------------------------------------------

def _emit(event: dict) -> None:
    """Safely call the current on_event callback (if one is set)."""
    on_event = _on_event_var.get()
    if on_event is not None:
        try:
            on_event(event)
        except Exception:
            pass


def create_initial_state(query: str) -> AgentState:
    """Build a fresh state for each agent execution."""
    return {
        "original_query": query,
        "sub_queries": [],
        "collected_docs": [],
        "fulltext_docs": [],
        "fulltext_done": False,
        "extension_docs": [],
        "experience_docs": [],
        "final_answer": "",
        "professor_query": None,
        "professor_list_query": None,
        "needs_sql_search": True,
        "needs_experience": False,
        "mentioned_school_ids": [],
        "mentioned_school_names": [],
        "verified_docs": [],
        "is_sufficient": True,
        "insufficiency_reason": "",
        "retry_count": 0,
        "generated_answer": False,
        "wants_recommendation": False,
        "profile": {},
        "recommend_docs": [],
    }


def set_execution_context(
    on_event: Optional[Callable[[dict], None]],
    cancel_event: Optional[threading.Event],
) -> None:
    _on_event_var.set(on_event)
    _cancel_event_var.set(cancel_event)


def clear_execution_context() -> None:
    _on_event_var.set(None)
    _cancel_event_var.set(None)
