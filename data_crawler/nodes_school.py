"""Main-graph nodes: Node 1 (init), Node 2 (BFS level by level), the Node 4 fan-out,
Node 9 (sufficiency), Node 10-14 (tagging / chunking / embedding / db_writer / finalize).
"""
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from langgraph.types import Send

from .config import CONFIG, SCHOOLS
from .state import SchoolState, ScrapeState
from .url_tools import normalize_url, application_scope_exclusion
from .browser import crawl_one_layer, scrape_url
from .text_clean import clean_noise
from .llm import call_llm_json
from .prompts import url_filter_prompt
from . import db as dbm

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
URL_RESULT_DIR = Path(__file__).resolve().parent / "url"

URL_FILTER_BATCH_SIZE = max(1, int(os.getenv("URL_FILTER_BATCH_SIZE", "40")))

# Important fields (used for the sufficiency coverage evaluation)
IMPORTANT_FIELDS = ["toefl_ibt_min", "ielts_min", "gre_required",
                    "application_fee_usd", "tuition_per_year_usd", "rec_letter_count"]


def _crawl_candidate_priority(item: dict) -> tuple:
    """Prioritize evidence pages when MAX_PAGES cannot cover every kept URL."""
    evidence = f"{item.get('url', '')} {item.get('anchor_text', '')}".lower()
    score = 0
    weighted_hints = (
        (("application-checklist", "application checklist", "graduate-requirement",
          "admission-requirement", "required-material", "/requirements"), 150),
        (("english-proficiency", "english proficiency", "toefl", "ielts", "duolingo"), 130),
        (("international-applicant", "international applicant"), 120),
        (("masters-admission", "master's admission", "ms admission"), 115),
        (("application-fee", "application fee", "fee-waiver", "fee waiver"), 110),
        (("tuition", "cost-of-attendance", "cost of attendance"), 105),
        (("deadline", "admission-calendar"), 100),
        (("admission", "apply", "application"), 80),
        (("faq", "frequently-asked"), 70),
        (("funding", "financial-aid", "fellowship", "scholarship"), 60),
    )
    for hints, weight in weighted_hints:
        if any(hint in evidence for hint in hints):
            score = max(score, weight)
    if any(hint in evidence for hint in (
            "joint-degree", "joint degree", "msmba", "ms-law",
            "bachelors-masters", "bachelor's/master")):
        score -= 150
    return (-score, item.get("root_index", 0), item.get("url", ""))


# ──────────────────────────────────────────────
# Node 1: init_crawl
# ──────────────────────────────────────────────

def init_crawl(state: SchoolState) -> dict:
    school_id = state["school_id"]
    school = next((s for s in SCHOOLS if s["school_id"] == school_id), None)
    if school is None:
        raise ValueError(f"school_id={school_id} not found in config/root_url.py")

    configured_roots = state.get("root_urls_override") or school["roots"]
    roots = [normalize_url(r) for r in configured_roots if r and r.startswith("http")]
    if not roots:
        raise ValueError(f"school_id={school_id} has no valid root URL")
    max_depth = state.get("max_depth")
    max_depth = CONFIG.MAX_DEPTH if max_depth is None else max_depth
    max_pages = state.get("max_pages")
    max_pages = CONFIG.MAX_PAGES if max_pages is None else max_pages

    university_id = None
    known_codes: list[str] = []
    skip_school = False

    if not state.get("dry_run"):
        conn = dbm.get_connection()
        try:
            dbm.ensure_migrations(conn)
            university_id = dbm.get_or_create_university(conn, school_id)
            known_codes = dbm.list_known_program_codes(conn, university_id)
            hours = state.get("skip_recent_hours") or 0
            if hours > 0 and dbm.school_recently_extracted(conn, university_id, hours):
                skip_school = True
                print(f"[SKIP] {school_id} -- data in the DB was updated within {hours}h; skipping the whole school")
        finally:
            conn.close()

    queue = [{"url": r, "depth": 0, "root_index": i} for i, r in enumerate(roots)]
    print(f"[INIT] {school_id}: {len(roots)} roots, max_depth={max_depth}, max_pages={max_pages}")

    return {
        "roots": roots,
        "university_id": university_id,
        "known_program_codes": known_codes,
        "skip_school": skip_school,
        "max_depth": max_depth,
        "max_pages": max_pages,
        "current_depth": 0,
        "url_queue": queue,
        "visited_urls": [q["url"] for q in queue],
        "discovered_urls": [],
        "dropped_urls": {},
        "external_urls": [],
        "total_crawled": 0,
        "url_filter_candidates": [],
        "url_filter_decisions": [],
        "unique_pages": [],
    }


def after_init(state: SchoolState) -> str:
    return "finalize_school" if state.get("skip_school") else "discover_urls"


# ──────────────────────────────────────────────
# Node 2: discover_urls (one level at a time, checkpointed at the end of each level)
# ──────────────────────────────────────────────

def discover_urls(state: SchoolState) -> dict:
    layer = state.get("url_queue", [])
    if not layer:
        return {}

    max_pages = state.get("max_pages", CONFIG.MAX_PAGES)
    total_crawled = state.get("total_crawled", 0)
    if max_pages > 0:
        layer = layer[: max(0, max_pages - total_crawled)]
    if not layer:
        return {"url_queue": []}

    depth = layer[0]["depth"]
    print(f"\n[BFS depth={depth}] {len(layer)} pages to crawl ({state['school_id']})")

    layer_results = crawl_one_layer(layer, state["roots"])

    visited = set(state.get("visited_urls", []))
    discovered = list(state.get("discovered_urls", []))
    discovered_set = set(discovered)
    dropped = dict(state.get("dropped_urls", {}))
    external = set(state.get("external_urls", []))
    root_index_by_url = {item["url"]: item["root_index"] for item in layer}

    candidates_by_url: dict[str, dict] = {}
    max_depth = state.get("max_depth", CONFIG.MAX_DEPTH)

    for url, dep, result in layer_results:
        total_crawled += 1
        if not result["error"] and url not in discovered_set:
            discovered.append(url)
            discovered_set.add(url)
        dropped.update(result["drop"])
        external.update(result["external"])

        for child_url in sorted(result["keep"]):
            if child_url not in visited:
                visited.add(child_url)
                if dep + 1 <= max_depth:
                    candidates_by_url[child_url] = {
                        "url": child_url,
                        "source_url": url,
                        "anchor_text": result.get("keep_anchors", {}).get(child_url, ""),
                        "depth": dep + 1,
                        "root_index": root_index_by_url.get(url, 0),
                    }

    # Drop statistics (reusing the level summary of crawl_tree)
    reasons: dict = {}
    for _, _, r in layer_results:
        for reason in r["drop"].values():
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        top = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"  [depth={depth} drop] " + "  ".join(f"{r}={c}" for r, c in top))

    return {
        "current_depth": depth + 1,
        "url_queue": [],
        "visited_urls": sorted(visited),
        "discovered_urls": discovered,
        "dropped_urls": dropped,
        "external_urls": sorted(external),
        "total_crawled": total_crawled,
        "url_filter_candidates": list(candidates_by_url.values()),
    }


def _normalize_url_filter_batch(candidates: list[dict], raw: object) -> list[dict]:
    """Turn the untrusted LLM JSON into one decision per candidate URL."""
    expected = {item["url"]: item for item in candidates}
    raw_items = raw.get("decisions", []) if isinstance(raw, dict) else []
    received: dict[str, list[dict]] = {}
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict) or item.get("url") not in expected:
            continue
        received.setdefault(item["url"], []).append(item)

    normalized = []
    for url, candidate in expected.items():
        items = received.get(url, [])
        raw_decisions = {str(item.get("decision", "")).strip().lower() for item in items}
        reason = next((str(item.get("reason", "")).strip() for item in items
                       if str(item.get("reason", "")).strip()), "")
        try:
            confidence = max(float(item.get("confidence", 0)) for item in items)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        fallback_reason = ""
        if not items:
            decision, fallback_reason = "keep", "The LLM did not return this URL; kept by the conservative policy"
        elif "keep" in raw_decisions:
            decision = "keep"
            if len(items) > 1:
                fallback_reason = "The LLM returned it more than once and at least one was keep; kept by the keep-first policy"
        elif raw_decisions != {"drop"}:
            decision, fallback_reason = "keep", "The LLM decision format is invalid; kept by the conservative policy"
        elif not reason:
            decision, fallback_reason = "keep", "The LLM gave no filtering reason; kept by the conservative policy"
        else:
            # When the decision/reason format is valid, respect the LLM's drop; confidence is only for the audit,
            # and an explicit drop must not be flipped to keep just because of low confidence.
            decision = "drop"

        normalized.append({
            **candidate,
            "decision": decision,
            "reason": fallback_reason or reason or "The LLM gave no reason; kept by the conservative policy",
            "confidence": confidence,
            "decision_source": "fallback" if fallback_reason else "llm",
        })
    return normalized


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    temp.replace(path)


def _write_url_filter_review(state: SchoolState, decisions: list[dict]) -> str:
    """Overwrite the full audit at every level and split it into files by keep/drop for manual inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{state['school_id']}_url_filter_review.json"
    kept = sum(1 for item in decisions if item["decision"] == "keep")
    payload = {
        "school_id": state["school_id"],
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total": len(decisions),
            "kept": kept,
            "dropped": len(decisions) - kept,
            "fallback_kept": sum(1 for item in decisions
                                 if item.get("decision_source") == "fallback"),
        },
        "decisions": decisions,
    }
    _write_json_atomic(out, payload)

    common = {
        "school_id": state["school_id"],
        "generated_at": payload["generated_at"],
    }
    keep_items = [item for item in decisions if item["decision"] == "keep"]
    drop_items = [item for item in decisions if item["decision"] == "drop"]
    _write_json_atomic(
        URL_RESULT_DIR / "keep" / f"{state['school_id']}.json",
        {**common, "decision": "keep", "count": len(keep_items), "urls": keep_items},
    )
    _write_json_atomic(
        URL_RESULT_DIR / "drop" / f"{state['school_id']}.json",
        {**common, "decision": "drop", "count": len(drop_items), "urls": drop_items},
    )
    # all_url deliberately stores only an array of URL strings: the full candidate list of root + keep/drop.
    all_urls = list(dict.fromkeys([
        *state.get("roots", []),
        *(item["url"] for item in decisions if item.get("url")),
    ]))
    _write_json_atomic(
        URL_RESULT_DIR / "all_url" / f"{state['school_id']}.json",
        all_urls,
    )
    return str(out)


def filter_discovered_urls(state: SchoolState) -> dict:
    """A batch LLM pre-screen of this level's new URLs, arranged into the next BFS queue and written to the audit."""
    candidates = state.get("url_filter_candidates", [])
    previous = list(state.get("url_filter_decisions", []))
    if not candidates:
        _write_url_filter_review(state, previous)
        return {"url_queue": [], "url_filter_candidates": []}

    current: list[dict] = []
    llm_candidates = []
    for item in candidates:
        exclusion = application_scope_exclusion(item["url"], item.get("anchor_text", ""))
        if exclusion:
            current.append({**item, "decision": "drop", "reason": exclusion,
                            "confidence": 1.0, "decision_source": "rule"})
        else:
            llm_candidates.append(item)

    for start in range(0, len(llm_candidates), URL_FILTER_BATCH_SIZE):
        batch = llm_candidates[start:start + URL_FILTER_BATCH_SIZE]
        try:
            raw = call_llm_json(url_filter_prompt(state["school_id"], state["roots"], batch))
            current.extend(_normalize_url_filter_batch(batch, raw))
        except Exception as exc:
            reason = f"The batch LLM judgment failed; kept by the conservative policy: {str(exc)[:200]}"
            current.extend([{**item, "decision": "keep", "reason": reason,
                             "confidence": 0.0, "decision_source": "fallback"}
                            for item in batch])

    kept = sorted(
        (item for item in current if item["decision"] == "keep"),
        key=_crawl_candidate_priority,
    )
    queue = [{"url": item["url"], "depth": item["depth"],
              "root_index": item["root_index"]}
             for item in kept]
    dropped = dict(state.get("dropped_urls", {}))
    for item in current:
        if item["decision"] == "drop":
            dropped[item["url"]] = f"llm:{item['reason']}"

    decisions = previous + current
    audit_file = _write_url_filter_review(state, decisions)
    print(f"  [URL FILTER] batches={(len(llm_candidates) + URL_FILTER_BATCH_SIZE - 1) // URL_FILTER_BATCH_SIZE} "
          f"keep={len(queue)} drop={len(current) - len(queue)} audit={audit_file}")
    return {
        "url_queue": queue,
        "dropped_urls": dropped,
        "url_filter_candidates": [],
        "url_filter_decisions": decisions,
    }


def should_continue_bfs(state: SchoolState) -> str:
    max_pages = state.get("max_pages", CONFIG.MAX_PAGES)
    if (state.get("url_queue")
            and (max_pages <= 0 or state.get("total_crawled", 0) < max_pages)):
        return "discover_urls"
    return "plan_scrape"


# ──────────────────────────────────────────────
# Node 4 fan-out: plan_scrape -(Send)-> scrape_page -> collect_scraped
# ──────────────────────────────────────────────

def plan_scrape(state: SchoolState) -> dict:
    return {}


def dispatch_scrape(state: SchoolState):
    already = {p["url"] for p in state.get("scraped_pages", [])}
    pending = [u for u in state.get("discovered_urls", []) if u not in already]
    if not pending:
        return "collect_scraped"
    print(f"\n[SCRAPE] fan-out {len(pending)} pages ({state['school_id']})")
    return [Send("scrape_page", {"school_id": state["school_id"], "url": url})
            for url in pending]


def scrape_page(state: ScrapeState) -> dict:
    """Node 4: full text of one page + structured_markdown / tables / content_hash / PDF."""
    url = state["url"]
    extracted = scrape_url(url)
    record = {"url": url, "requested_url": url, **extracted}
    status = f"[WARN] {extracted['error'][:60]}" if extracted["error"] else f"{len(extracted['full_text'])} chars"
    print(f"  [scraped] {url} → {status}")
    return {"scraped_pages": [record]}


def collect_scraped(state: SchoolState) -> dict:
    """Deduplicate by final URL and filter out error pages; the content hash is only for diagnosis and never drops data across URLs.

    Different official pages can end up with the same or an empty content_hash because of a shared template, a failed load,
    or an old checkpoint that lacks the hash. Deduplicating across URLs by hash directly would lose admissions requirements,
    so only an identical final URL (including a redirect alias) is skipped.
    """
    first_url_by_hash: dict[str, str] = {}
    seen_urls: set[str] = set()
    unique = []

    for rec in state.get("scraped_pages", []):
        if rec.get("error") or not rec.get("full_text"):
            continue
        requested_url = rec.get("requested_url") or rec["url"]
        effective_url = normalize_url(rec.get("final_url") or rec["url"])
        if effective_url in seen_urls:
            print(f"  [SKIP] redirect alias: {requested_url[:70]} -> {effective_url[:70]}")
            continue
        seen_urls.add(effective_url)
        rec["requested_url"] = requested_url
        rec["url"] = effective_url
        rec["final_url"] = effective_url

        h = rec.get("content_hash")
        if h:
            first_url = first_url_by_hash.get(h)
            if first_url and first_url != effective_url:
                print(f"  [WARN] same content hash retained: {effective_url[:70]} "
                      f"(same as {first_url[:70]})")
            else:
                first_url_by_hash[h] = effective_url
        else:
            print(f"  [WARN] missing content hash retained: {effective_url[:80]}")
        unique.append(rec)

    print(f"[COLLECT] {len(unique)} pages to classify (after URL dedup / error removal)")
    return {"unique_pages": unique}


def dispatch_process(state: SchoolState):
    pages = state.get("unique_pages", [])
    if not pages:
        return "sufficiency_evaluator"
    return [Send("process_page", {
        "school_id": state["school_id"],
        "university_id": state.get("university_id"),
        "known_program_codes": state.get("known_program_codes", []),
        "page": page,
    }) for page in pages]


# ──────────────────────────────────────────────
# Node 9: sufficiency_evaluator (only produces the coverage report)
# ──────────────────────────────────────────────

def _coverage_report(page_results: list[dict]) -> dict:
    coverage: dict = {}
    for r in page_results:
        if r.get("status") != "ok":
            continue
        ext = r.get("extraction") or {}
        for prog in ext.get("programs", []):
            code = prog.get("program_code") or "?"
            got = coverage.setdefault(code, {"fields": set(), "deadlines": 0})
            got["fields"].update(prog.get("fields", {}).keys())
        for d in ext.get("deadlines", []):
            code = d.get("program_code") or "?"
            coverage.setdefault(code, {"fields": set(), "deadlines": 0})["deadlines"] += 1

    report = {}
    for code, got in coverage.items():
        report[code] = {
            "fields_extracted": sorted(got["fields"]),
            "deadline_count": got["deadlines"],
            "missing_important": [
                f for f in IMPORTANT_FIELDS
                if f not in got["fields"]
                and not (f == "toefl_ibt_min" and "toefl_min" in got["fields"])
            ],
        }
    return report


def _registrable_domain(netloc: str) -> str:
    parts = netloc.lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def sufficiency_evaluator(state: SchoolState) -> dict:
    report = _coverage_report(state.get("page_results", []))
    missing = sorted({field for item in report.values()
                      for field in item.get("missing_important", [])})
    if not report:
        missing_summary = "No structured extraction results at all"
    elif missing:
        missing_summary = "Missing: " + ", ".join(missing)
    else:
        missing_summary = "All important fields are present"
    result = {
        "sufficient": bool(report) and not missing,
        "missing_summary": missing_summary,
    }
    assessments = [
        r.get("target_assessment", {}).get("masters_available")
        for r in state.get("page_results", [])
        if r.get("target_assessment", {}).get("masters_available") is not None
    ]
    result["international_cs_masters_available"] = (
        True if any(value is True for value in assessments)
        else False if any(value is False for value in assessments)
        else None
    )
    print(f"[SUFFICIENCY] sufficient={result['sufficient']} "
          f"missing={result['missing_summary'][:100]}")

    return {
        "sufficiency_report": {"coverage": report, **result},
    }


# ──────────────────────────────────────────────
# Node 10-12: tagging / chunking / embedding
# ──────────────────────────────────────────────

def tagging(state: SchoolState) -> dict:
    """Rule-based tags: classification type + program codes (no LLM needed)."""
    for r in state.get("page_results", []):
        if r.get("status") != "ok":
            continue
        tags = [t["type"] for t in r.get("passed_types", [])]
        tags += [p["program_code"] for p in r.get("program_codes", []) if p.get("program_code")]
        r["tags"] = sorted(set(tags))
    return {}


def chunking(state: SchoolState) -> dict:
    """Cut chunks from structured_markdown (cleaned with the keyword list in config/cleaning.py)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )

    chunks = []
    for r in state.get("page_results", []):
        if r.get("status") != "ok":
            continue
        text = r.get("structured_markdown") or r.get("full_text") or ""
        text = clean_noise(text)
        if not text.strip():
            continue
        type_label = ",".join(t["type"] for t in r.get("passed_types", [])) or "general"
        prefix = f"[{state['school_id']} | {type_label}] "
        for piece in splitter.split_text(text):
            if len(piece.strip()) < 50:
                continue
            chunks.append({"url": r["url"], "text": prefix + piece.strip()})

    print(f"[CHUNK] {len(chunks)} chunks in total")
    return {"chunks": chunks}


def embedding(state: SchoolState) -> dict:
    """The ENABLE_EMBEDDING switch (off by default; relies on fts_vector full-text search).

    When on, it uses the backend embedder's model (BAAI/bge-m3, 1024 dimensions;
    the env var BGE_EMBED_MODEL_PATH can point to a local path).
    """
    if not state.get("enable_embedding"):
        print("[EMBED] ENABLE_EMBEDDING=off, skipping (it can be backfilled later with backfill_embeddings.py)")
        return {}

    from sentence_transformers import SentenceTransformer
    model_path = os.getenv("BGE_EMBED_MODEL_PATH", "BAAI/bge-m3")
    model = SentenceTransformer(model_path)

    chunks = state.get("chunks", [])
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    for c, v in zip(chunks, vectors):
        c["embedding"] = v.tolist()
    print(f"[EMBED] Done: {len(chunks)} chunks")
    return {"chunks": chunks}


# ──────────────────────────────────────────────
# Node 13: db_writer
# ──────────────────────────────────────────────

def _validate_school_write_scope(state: SchoolState, ok_results: list[dict]) -> None:
    """Stop a school-A URL from being written into school B when the checkpoint/state gets mixed up."""
    school_id = state["school_id"]
    school = next((item for item in SCHOOLS if item["school_id"] == school_id), None)
    if not school:
        raise RuntimeError(f"DB write aborted: no root configuration found for school_id={school_id}")

    allowed_domains = {
        _registrable_domain(urlparse(root).netloc)
        for root in school.get("roots", []) if root and root.startswith("http")
    }
    if not allowed_domains:
        raise RuntimeError(f"DB write aborted: school_id={school_id} has no valid root domain")

    scoped_urls = [
        *(state.get("roots") or []),
        *(result.get("url", "") for result in ok_results),
        *(chunk.get("url", "") for chunk in state.get("chunks", [])),
    ]
    invalid = [url for url in scoped_urls
               if url and _registrable_domain(urlparse(url).netloc) not in allowed_domains]
    if invalid:
        samples = ", ".join(invalid[:5])
        raise RuntimeError(
            f"DB write aborted: the state of {school_id} contains domains of other schools; "
            f"allowed={sorted(allowed_domains)}, offending URLs={samples}"
        )


def _common_source_priority(result: dict) -> int:
    """On a conflict within the same scope, write the weak source first and the strong source last, so the DB finally keeps the strong evidence."""
    text = f"{result.get('url', '')} {result.get('title', '')}".lower()
    if "checklist" in text:
        return 30
    if "faq" in text:
        return 40
    if "requirement" in text or "admission" in text:
        return 50
    if "english" in text or "language" in text:
        return 35
    return 10


def _program_write_priority(result: dict) -> tuple[int, int]:
    """Write the weak scope first and the CS master's program-specific data last."""
    scope_rank = {
        "school_wide": 0,
        "department_wide": 1,
        "program_specific": 2,
    }
    return (scope_rank.get(result.get("scope"), 0), _common_source_priority(result))


def db_writer(state: SchoolState) -> dict:
    school_id = state["school_id"]
    ok_results = [r for r in state.get("page_results", []) if r.get("status") == "ok"]
    _validate_school_write_scope(state, ok_results)

    # Whether or not it is a dry run, write a JSON that can be audited by hand. In DB mode we cannot only put
    # the original text into web_pages, or without a DB connection we could not tell whether the crawler or the LLM missed data.
    out = OUTPUT_DIR / f"{school_id}_result.json"
    payload = {
        "school_id": school_id,
        "generated_at": datetime.now().isoformat(),
        "page_results": [{k: v for k, v in r.items()
                          if k not in ("full_text", "structured_markdown")} | {
                             "data": r.get("full_text", ""),
                             "char_count": len(r.get("full_text", ""))}
                         for r in state.get("page_results", [])],
        "review_items": state.get("review_items", []),
        "url_filter_decisions": state.get("url_filter_decisions", []),
        "sufficiency_report": state.get("sufficiency_report", {}),
        "chunk_count": len(state.get("chunks", [])),
    }
    _write_json_atomic(out, payload)
    print(f"[OUTPUT] Full result and original text written to {out}")

    if state.get("dry_run"):
        return {"summary": {"db": "dry_run", "output_file": str(out)}}

    conn = dbm.get_connection()
    # Do not trust a university_id in the checkpoint that may be stale; resolve it again from school_id on every write.
    university_id = dbm.get_or_create_university(conn, school_id)
    stale_id = state.get("university_id")
    if stale_id is not None and stale_id != university_id:
        print(f"  [WARN] stale university_id corrected: state={stale_id} db={university_id} ({school_id})")
    stats = {"pages": 0, "programs": 0, "global_extractions": 0,
             "deadlines": 0, "scholarships": 0,
             "materials": 0, "evidence": 0, "review": 0, "chunks": 0}
    chunks_by_url: dict[str, list] = {}
    for c in state.get("chunks", []):
        chunks_by_url.setdefault(c["url"], []).append(c)
    # If the same fact has already passed validation on another, more reliable page, it should not stay in review_queue.
    accepted_program_values = {
        (
            prog.get("program_code"),
            field_name,
            str(field_value.get("value")),
        )
        for result in ok_results
        for prog in (result.get("extraction") or {}).get("programs", [])
        for field_name, field_value in (prog.get("fields") or {}).items()
        if isinstance(field_value, dict) and field_value.get("value") is not None
    }

    try:
        program_id_by_code: dict[str, int] = {}

        # Stage 1: first collect the programs that have official heading evidence, so the school-wide form dropdowns do not
        # create empty shells by mistake. Then write the SCHOOL_WIDE shared fields first; later page values can overwrite them.
        verified_meta: dict[str, dict] = {}
        school_common_results = []
        department_common_results = []
        for result in ok_results:
            for meta in result.get("program_codes", []):
                code = meta.get("program_code")
                if code == "SCHOOL_WIDE":
                    school_common_results.append(result)
                elif code == "CS_DEPARTMENT_WIDE":
                    department_common_results.append(result)
                elif code and meta.get("official_evidence"):
                    verified_meta.setdefault(code, meta)

        # Final fallback: if no official name was recognized but the page does not clearly rule out a terminal
        # master's, still create only the single current international CS master's target; never fall back to the old multi-program codes.
        target_explicitly_unavailable = any(
            result.get("target_assessment", {}).get("masters_available") is False
            for result in ok_results
        )
        if not verified_meta and not target_explicitly_unavailable:
            verified_meta["INTERNATIONAL_CS_MASTERS"] = {
                "program_code": "INTERNATIONAL_CS_MASTERS",
                "degree_type": "MS",
                "program_name": "International Computer Science Master's",
                "department": "Computer Science",
                "official_evidence": "default international CS master's target",
                "is_default": True,
            }
            print(f"  [program fallback] {school_id} -> created INTERNATIONAL_CS_MASTERS")
        elif not verified_meta:
            print(f"  [program unavailable] {school_id} -> not creating a fictional international CS master's target")

        for code, meta in verified_meta.items():
            program_id_by_code[code] = dbm.upsert_program(
                conn, university_id, code, meta, {}, "", None,
            )

        # Priority: school-wide fallback < CS/CSE department; a checklist at the same level is strongest.
        common_results = [
            *sorted(school_common_results, key=_common_source_priority),
            *sorted(department_common_results, key=_common_source_priority),
        ]
        for result in common_results:
            for prog in (result.get("extraction") or {}).get("programs", []):
                if prog.get("program_code") not in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE"):
                    continue
                fields = {name: value.get("value")
                          for name, value in (prog.get("fields") or {}).items()
                          if isinstance(value, dict)}
                for code, meta in verified_meta.items():
                    program_id_by_code[code] = dbm.upsert_program(
                        conn, university_id, code, meta, fields,
                        result["url"], result.get("confidence"),
                    )

        # Under a single target code, conflicts are resolved by scope:
        # university-wide < CS/CSE department < CS master's program.
        for r in sorted(ok_results, key=_program_write_priority):
            ext = r.get("extraction") or {}
            meta_by_code = {p["program_code"]: p for p in r.get("program_codes", [])
                            if p.get("program_code") in verified_meta}

            # programs (overwritten only with non-null values)
            for prog in ext.get("programs", []):
                code = prog.get("program_code")
                if not code or code in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE") or code not in verified_meta:
                    continue
                fields = {name: fv.get("value") for name, fv in (prog.get("fields") or {}).items()}
                pid = dbm.upsert_program(conn, university_id, code,
                                         meta_by_code.get(code, {}), fields,
                                         r["url"], r.get("confidence"))
                program_id_by_code[code] = pid
                stats["programs"] += 1

            # Pages with no field extraction but with an identified program still get a program skeleton
            for code, meta in meta_by_code.items():
                if code not in program_id_by_code:
                    pid = dbm.upsert_program(conn, university_id, code, meta, {},
                                             r["url"], r.get("confidence"))
                    program_id_by_code[code] = pid

            main_program_id = None
            codes = [p.get("program_code") for p in r.get("program_codes", [])]
            if codes and codes[0] in program_id_by_code:
                main_program_id = program_id_by_code[codes[0]]

            # web_pages (cleaned full text)
            raw_text = clean_noise(r.get("full_text", ""))
            page_id = dbm.upsert_web_page(conn, university_id, r["url"],
                                          r.get("passed_types", []), raw_text,
                                          main_program_id)
            stats["pages"] += 1

            # School-wide / CS-department extraction results are stored on their own. Even when there is no official program yet,
            # they can still be audited from the DB, and re-applied once a program exists.
            global_codes = [p for p in r.get("program_codes", [])
                            if p.get("program_code") in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE")]
            if global_codes and ext:
                for meta in global_codes:
                    code = meta["program_code"]
                    scoped_extraction = {
                        "programs": [p for p in ext.get("programs", [])
                                     if p.get("program_code") == code],
                        "deadlines": [p for p in ext.get("deadlines", [])
                                      if p.get("program_code") == code],
                        "scholarships": [p for p in ext.get("scholarships", [])
                                         if p.get("program_code") == code],
                        "app_materials": [p for p in ext.get("app_materials", [])
                                          if p.get("program_code") == code],
                        "evidence_paragraphs": [
                            p for p in ext.get("evidence_paragraphs", [])
                            if p.get("program_code") == code
                        ],
                    }
                    if any(scoped_extraction.values()):
                        dbm.upsert_global_extraction(
                            conn, university_id, page_id,
                            meta.get("scope") or r.get("scope") or "school_wide",
                            code, scoped_extraction, r.get("confidence"), r["url"],
                        )
                        stats["global_extractions"] += 1

            # Child tables: look up, then decide insert/update (v4 3-1); filter out empty-shell rows with no real content
            for d in ext.get("deadlines", []):
                pid = program_id_by_code.get(d.get("program_code"))
                if pid and (d.get("deadline_date") or d.get("application_open_date") or d.get("application_close_date")
                            or d.get("decision_release_date")
                            or (d.get("semester") and d.get("note"))):
                    dbm.upsert_deadline(conn, pid, d)
                    stats["deadlines"] += 1
            for s in ext.get("scholarships", []):
                pid = program_id_by_code.get(s.get("program_code"))
                if pid and s.get("name"):
                    dbm.upsert_scholarship(conn, pid, s, r["url"])
                    stats["scholarships"] += 1
            for m in ext.get("app_materials", []):
                pid = program_id_by_code.get(m.get("program_code"))
                if pid and m.get("material_type") and (m.get("requirement") or m.get("note")):
                    dbm.upsert_app_material(conn, pid, m)
                    stats["materials"] += 1
            for evidence in ext.get("evidence_paragraphs", []):
                pid = program_id_by_code.get(evidence.get("program_code"))
                if pid and evidence.get("evidence_text"):
                    dbm.upsert_program_evidence(conn, pid, evidence, r["url"])
                    stats["evidence"] += 1

            # document_chunks
            page_chunks = chunks_by_url.get(r["url"], [])
            if page_chunks:
                dbm.replace_chunks(conn, university_id, page_id, school_id, r["url"],
                                   r.get("passed_types", []), page_chunks, main_program_id)
                stats["chunks"] += len(page_chunks)

            # review_queue (the fields that failed validation on this page)
            for item in state.get("review_items", []):
                if item.get("url") == r["url"]:
                    accepted_key = (
                        item.get("program_code"),
                        item.get("field_name"),
                        str(item.get("field_value")),
                    )
                    if item.get("field_value") is not None \
                            and accepted_key in accepted_program_values:
                        continue
                    dbm.insert_review_item(conn, university_id, page_id, item)
                    stats["review"] += 1

        # School-wide deadlines / scholarships / materials are also distributed to the confirmed programs.
        for result in common_results:
            ext = result.get("extraction") or {}
            for code, pid in program_id_by_code.items():
                for item in ext.get("deadlines", []):
                    if item.get("program_code") in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE") and (
                            item.get("deadline_date") or item.get("application_open_date") or item.get("application_close_date")
                            or item.get("decision_release_date")
                            or (item.get("semester") and item.get("note"))):
                        dbm.upsert_deadline(conn, pid, {**item, "program_code": code})
                        stats["deadlines"] += 1
                for item in ext.get("scholarships", []):
                    if item.get("program_code") in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE") and item.get("name"):
                        dbm.upsert_scholarship(conn, pid, {**item, "program_code": code}, result["url"])
                        stats["scholarships"] += 1
                for item in ext.get("app_materials", []):
                    if item.get("program_code") in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE") and item.get("material_type") and (
                            item.get("requirement") or item.get("note")):
                        dbm.upsert_app_material(conn, pid, {**item, "program_code": code})
                        stats["materials"] += 1
                for item in ext.get("evidence_paragraphs", []):
                    if (item.get("program_code") in ("SCHOOL_WIDE", "CS_DEPARTMENT_WIDE")
                            and item.get("evidence_text")):
                        dbm.upsert_program_evidence(
                            conn, pid, {**item, "program_code": code}, result["url"]
                        )
                        stats["evidence"] += 1
    finally:
        conn.close()

    print(f"[DB] {school_id}: {stats}")
    return {"summary": {"db": stats, "output_file": str(out)}}


# ──────────────────────────────────────────────
# Node 14: finalize_school
# ──────────────────────────────────────────────

def finalize_school(state: SchoolState) -> dict:
    results = state.get("page_results", [])
    ok = sum(1 for r in results if r.get("status") == "ok")
    dropped = sum(1 for r in results if r.get("status") == "dropped")
    summary = {
        "school_id": state["school_id"],
        "skipped": bool(state.get("skip_school")),
        "urls_discovered": len(state.get("discovered_urls", [])),
        "urls_llm_kept": sum(1 for item in state.get("url_filter_decisions", [])
                             if item.get("decision") == "keep"),
        "urls_llm_dropped": sum(1 for item in state.get("url_filter_decisions", [])
                                if item.get("decision") == "drop"),
        "pages_kept": ok,
        "pages_dropped": dropped,
        "review_items": len(state.get("review_items", [])),
        "chunks": len(state.get("chunks", [])),
        "finished_at": datetime.now().isoformat(),
    }
    print(f"\n{'='*60}\n[FINISH] {json.dumps(summary, ensure_ascii=False, indent=2)}\n{'='*60}")
    return {"summary": summary}
