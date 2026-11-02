"""LangGraph state definitions.

SchoolState  -- the main graph (one thread_id per school, BFS level by level + fan-out)
ScrapeState  -- the single-page input of the Node 4 fan-out
ProcessState -- the state of the Node 5-8 page subgraph (one page: classify -> map to a program -> extract -> validate)
"""
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


def merge_dict(a: Optional[dict], b: Optional[dict]) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class SchoolState(TypedDict, total=False):
    # -- Basics ----------------------------------------------
    school_id: str
    university_id: Optional[int]
    roots: list[str]
    root_urls_override: list[str]  # CLI --root-url; when set, overrides the school's default roots
    max_depth: int
    max_pages: int
    skip_school: bool          # the Node 1 SKIP_RECENT decision
    dry_run: bool              # do not write the DB (for local testing)
    enable_embedding: bool     # Node 12 switch (off by default)
    skip_recent_hours: int     # when > 0 enables SKIP_RECENT (checks last_extracted_at in the DB)
    known_program_codes: list[str]  # program_codes already in the DB (a reference for Node 6)

    # -- Phase 1: BFS (level by level, follows the checkpoint) ----
    current_depth: int
    url_queue: list[dict]      # [{"url", "depth", "root_index"}] the URLs to crawl on the current level
    visited_urls: list[str]    # URLs already queued (a resume does not re-crawl them)
    discovered_urls: list[str] # URLs that passed the filter and go to Phase 2
    dropped_urls: dict         # url -> reason
    external_urls: list[str]
    total_crawled: int
    url_filter_candidates: list[dict]  # this level's URLs for the LLM to judge + anchor/source/depth
    url_filter_decisions: list[dict]   # accumulated LLM decisions, output for manual inspection

    # -- Phase 2: collecting the scrape fan-out -----------------
    scraped_pages: Annotated[list[dict], operator.add]
    unique_pages: list[dict]   # after final-URL dedup and error removal

    # -- Phase 3-4: collecting the page-pipeline fan-out --------
    page_results: Annotated[list[dict], operator.add]
    review_items: Annotated[list[dict], operator.add]

    # -- Phase 5: sufficiency -----------------------------
    sufficiency_report: dict

    # ── Phase 6 ──────────────────────────────────────────
    chunks: list[dict]
    summary: Annotated[dict, merge_dict]


class ScrapeState(TypedDict, total=False):
    """Single-page input sent to scrape_page."""
    school_id: str
    url: str
    depth: int


class ProcessState(TypedDict, total=False):
    """State of the page subgraph (Node 5-8)."""
    school_id: str
    university_id: Optional[int]
    page: dict                 # one entry of unique_pages (with full_text / structured_markdown ...)
    known_program_codes: list[str]
    scope: str                 # "page" | "school_wide"

    # Node 5
    classification: dict       # {"is_relevant": bool, "types": [{"type","confidence"}], "reason"}

    # Node 6
    program_codes: list[dict]  # [{"program_code","degree_type","program_name","department"}]
    target_assessment: dict    # whether an international CS master's exists and its scope of application

    # Node 7
    extraction: dict           # programs/deadlines/scholarships/app_materials/evidence_paragraphs
    extraction_retries: int

    # Node 8
    validation: dict           # {"passed": bool, "issues": [...], "confidence": float}
    validation_repair_retries: int
    rejected_issues: list[dict]  # issues of earlier validation rounds, whose values the repair step removed

    # Output (merged back into the main graph)
    page_results: Annotated[list[dict], operator.add]
    review_items: Annotated[list[dict], operator.add]


# Page classification types (the same 6+2 list as agreed; faculty / other are dropped after classification)
KEEP_TYPES = ["admissions", "deadlines", "program", "tuition", "scholarship", "faq"]
DROP_TYPES = ["faculty", "other"]
