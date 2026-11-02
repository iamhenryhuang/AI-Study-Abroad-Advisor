# data_crawler

LangGraph pipeline that crawls a university's official pages, uses a language model to judge which pages matter and to extract admission fields, and checks every extracted value against the page text before it is stored. Installation and database setup are in the [top-level README](../README.md).

## Contents

1. [Usage](#1-usage)
2. [Graph](#2-graph)
3. [Page classification](#3-page-classification)
4. [Language model providers](#4-language-model-providers)
5. [Database behavior](#5-database-behavior)
6. [Settings](#6-settings)
7. [Known limitations](#7-known-limitations)

## 1. Usage

Run from the repository root.

```bash
# Try a small crawl first, without touching the database
python -m data_crawler.main --school-id purdue --max-depth 1 --max-pages 10 --dry-run

# Real run: writes to the database with a PostgresSaver checkpoint (thread_id = school_id)
python -m data_crawler.main --school-id stanford --max-depth 2

# Crawl one root only (overrides the roots in config/root_url.py for this run)
python -m data_crawler.main --school-id stanford --root-url https://www.cs.stanford.edu/ --max-depth 2

# Resume after an interruption (BFS checkpoints after every level)
python -m data_crawler.main --school-id stanford --resume

# Other switches
#   --skip-recent-hours 24   skip the school if the database was extracted within 24 h
#   --enable-embedding       also compute embeddings (off by default; full-text search needs none)

# Backfill embeddings separately
python -m data_crawler.backfill_embeddings [--school-id stanford]
```

`--dry-run` does not touch the database. It writes `data_crawler/output/{school_id}_result.json` and uses an in-memory checkpointer.

## 2. Graph

```text
Main graph (SchoolState, thread_id = school_id, PostgresSaver)
  Node 1   init_crawl             find or create universities.id; skip if recently extracted
  Node 2   discover_urls (loop)   BFS one level at a time, checkpoint after every level;
                                  visited URLs live in state, so a resume does not re-crawl
  Node 3   URL filter             batch language-model judgment of the level's candidate URLs
  Node 4   scrape_page            fan-out, one branch per page, limited by NUM_WORKERS
           collect_scraped        final-URL deduplication, drops error pages and off-site redirects
  Node 5-8 process_page           fan-out into the page subgraph (below)
  Node 9   sufficiency_evaluator  rule-based coverage report
  Node 10  tagging                rule-based type and program codes
  Node 11  chunking               split structured markdown, clean with the keyword list
  Node 12  embedding              optional (ENABLE_EMBEDDING), BGE-M3
  Node 13  db_writer
  Node 14  finalize_school

Page subgraph (ProcessState)
  Node 5   content_classification   multi-label language-model classification
     irrelevant, faculty, or low confidence -> discard_page
  Node 6   identify_programs        decides whether the page applies to the international CS master's
  Node 7   structured_extraction    extracts fields from structured markdown, each with a source_excerpt
  Node 8   hallucination_validation compares field by field: the excerpt must be grounded in the page text
                                    (found verbatim, or at least 55% of its meaningful words occur on the page),
                                    and numbers or dates must appear inside the excerpt
           semantic_repair          removes rejected values, keeps their grounded context in program_evidence, and
                                    asks the language model to correct or fill fields; runs once or twice, then back to Node 8
           finalize_page            a rejected value goes to review_queue unless a grounded value replaced it
```

Node 3 prefers pages about English-language thresholds, checklists, international applicants, application fees and deadlines; it down-weights joint-degree and BS/MS pages and excludes current-student and current-fellowship pages.

## 3. Page classification

Multi-label, with six content types plus two that are discarded.

| Type | Meaning | Handling |
|---|---|---|
| admissions | requirements, process, eligibility | extracted |
| deadlines | deadlines and timelines | extracted |
| program | program, course and degree requirements | extracted |
| tuition | tuition and cost | extracted |
| scholarship | scholarships and financial aid | extracted |
| faq | frequently asked questions | extracted |
| faculty | professor and staff pages | discarded after classification |
| other | no admissions information | discarded |

## 4. Language model providers

The interface is an `openai.OpenAI` singleton configured from `backend/.env`. Set `LLM_PROVIDER=openai|groq|gemini`, or leave it unset to use the first provider whose key is present:

1. `OPENAI_API_KEY` uses OpenAI (`OPENAI_MODEL`, default `gpt-4.1`).
2. `GROQ_API_KEY` uses Groq's OpenAI-compatible endpoint (`GROQ_MODEL`, default `llama-3.3-70b-versatile`).
3. `GOOGLE_API_KEY` uses Gemini's OpenAI-compatible endpoint (`GEMINI_MODEL`, default `gemini-2.5-flash`).

Structured calls use JSON mode with temperature 0. Concurrency is limited by a semaphore (`LLM_MAX_CONCURRENCY`, default 2), and a 429 response triggers a backoff that follows the API's suggested delay, up to five attempts.

## 5. Database behavior

- **Migrations are additive and idempotent** (`db.py`, run automatically in Node 1). They only `CREATE TABLE IF NOT EXISTS` and `ADD COLUMN IF NOT EXISTS`, and never drop or clear data.
- **Child tables** use select-then-insert-or-update: deadlines by `(program_id, deadline_type, semester)`, scholarships by `(program_id, name)`, materials by `(program_id, material_type)`, writing `updated_at` on update.
- **Program columns** are overwritten only with non-null values, so a missing value never erases an earlier one.
- **Failed validation** never reaches the official tables. The field goes to `review_queue` (`reason=hallucination_detected`) with a `status` of pending, resolved or rejected for manual review.
- **Values that cannot be normalized** safely (dates, GPAs, language scores, conditional rules) are stored with their context in `program_evidence` (category, field name, excerpt, source URL) for retrieval.
- **TOEFL** old scale (0-120) and the scale in effect from 2026-01-21 (1-6) are stored separately in `toefl_ibt_min` and `toefl_ibt_new_scale_min`, with no conversion.

## 6. Settings

Everything the crawler is tuned by lives in the `data_crawler/config/` package:

| File | Content |
|---|---|
| `config/parameter.py` | `CONFIG`, page hints, thresholds, user agent |
| `config/blacklist.py` | ignored file extensions |
| `config/root_url.py` | per-school names and root URLs (`SCHOOLS`) |
| `config/cleaning.py` | keyword list used to clean page text |

To add a school, add its name and roots to `config/root_url.py`, and add its aliases to `SCHOOL_ALIASES` in `backend/retriever/school_aliases.py` so the agent can recognize it.

## 7. Known limitations

1. **Faculty pages are not processed.** They are classified and discarded; the `professors` table is filled from a faculty list you provide instead.
2. **One target model.** Only `INTERNATIONAL_CS_MASTERS` is created per school. University-level and department-level rules are merged into one record, and no program is invented when an official CS page says there is no terminal master's.
3. **Re-running a thread accumulates state.** LangGraph accumulator channels cannot be reset, so a run without `--resume` on an existing thread opens a new `{school_id}-{timestamp}` thread and prints the `--resume --thread-id` command that continues it.
4. **Rate limits.** Free-tier providers can throttle a large fan-out. Concurrency limits and backoff recover automatically, but overall speed is bounded by the quota.
5. **Deadlines.** A date without an explicit year is not guessed; it is stored in `program_evidence`. Duplicates across pages and extended deadlines still need school-level merging.
6. **No `robots.txt` check.** Respect each site's terms of use.
7. **PDFs.** `config/blacklist.py` blocks `.pdf` in the BFS stage, so the PDF branch only applies to seed URLs or when `filter_url(allow_pdf=True)` is used.
8. **Checkpoint granularity.** BFS checkpoints per level. The scrape and process fan-outs are single supersteps: branches that finished before an interruption keep their results, and unfinished branches run again.
