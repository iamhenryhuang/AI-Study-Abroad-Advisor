# backend

FastAPI service and the LangGraph agent. It decomposes a question, routes it to the right sources, checks the evidence, and answers or abstains. Installation and database setup are in the [top-level README](../README.md).

## Contents

1. [API](#1-api)
2. [Retrieval paths](#2-retrieval-paths)
3. [Quality control](#3-quality-control)
4. [Faculty lookup](#4-faculty-lookup)
5. [When the language model is unavailable](#5-when-the-language-model-is-unavailable)
6. [Command line](#6-command-line)
7. [Layout](#7-layout)

## 1. API

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Run the agent and stream progress and the answer over SSE |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/experiences` | Submit a user's own application experience |
| `GET` | `/api/experiences` | Query user-submitted experiences |

`/api/chat` emits these event types: `thinking`, `tool_call`, `tool_result`, `llm_call`, `answer_chunk`, `answer`, `error`.

## 2. Retrieval paths

The planner decides which of these a question needs, and independent paths run in parallel.

| Path | Implementation | Source |
|---|---|---|
| SQL | `retriever/sql_search.py` | `programs` and its child tables (official requirements) |
| Hybrid | `retriever/hybrid_search.py` | embedding + full-text + rank fusion over `document_chunks` |
| Full-text | `retriever/fulltext_search.py` | `document_chunks.fts_vector` |
| Experience | `retriever/applicant_search.py` | `applicant_reports` and `user_experiences` (unofficial) |
| Recommend | `retriever/recommend.py` | community cases and the user's scores |
| Professor | `retriever/professor_list_search.py`, `professor_fetcher/` | `professors`; SerpAPI for a named professor |

Official-requirement questions try text-to-SQL first. When the result is insufficient, or the language model is unavailable, the agent searches the official page text. Embeddings are optional: without them the system uses PostgreSQL full-text search.

To enable vector retrieval, backfill embeddings once:

```bash
python -m data_crawler.backfill_embeddings --batch-size 16
```

To skip loading the embedding and reranker models, set this in `backend/.env`:

```env
ENABLE_HYBRID_SEARCH=false
```

## 3. Quality control

- **Verifier.** Checks whether the retrieved evidence covers each point the question asks for.
- **Full-text fallback.** When structured fields are not enough, searches the official page text.
- **Refiner.** When evidence is insufficient but something was retrieved, rewrites the query and retries at most once.
- **Abstain.** If coverage is still insufficient, the final step says so instead of generating an answer.
- **Critic.** Runs only when the language model actually generated an answer, and appends a warning to statements the retrieved evidence does not support. It never triggers another search.

Community experiences are unofficial. Answers label them as such and note sample bias; they should not be read as admission thresholds or guarantees.

## 4. Faculty lookup

Two kinds of faculty questions:

1. "Which Stanford professors work on AI?" is answered from the `professors` table you loaded.
2. "What does Fei-Fei Li research?" looks the named professor up through SerpAPI at query time.

Faculty rows attach to `universities.school_id`, so crawl the schools first. Build `db/data/professors.json` (the format is in [`db/data/README.md`](../db/data/README.md)), then run:

```bash
python backend/scripts/run.py load-professors
```

If the loader reports that a school does not exist, run `verify-db`, and make sure `init-schema` was not run again after crawling.

## 5. When the language model is unavailable

After the first permanent quota error, a process-level circuit breaker stops further OpenAI requests so that every agent node does not return the same 429. The agent then:

- recognizes schools, professors, experience and recommendation intents with local rules;
- searches `document_chunks` when SQL is unavailable;
- lets the Verifier accept the retrieved data as is;
- returns database excerpts with their sources instead of a generated answer, and skips the Critic.

Restarting the process makes the first model call re-check whether the provider is available.

## 6. Command line

```bash
python backend/scripts/run.py search "Gatech TOEFL 要求"
python backend/scripts/run.py rag "Purdue IELTS 要求"
python backend/scripts/run.py agent "Stanford 有哪些 AI 教授？"
python backend/scripts/run.py verify-db
```

Use `agent` for the real behavior. `search` exercises text-to-SQL only, and `rag` is a simpler single-pass retrieval flow.

## 7. Layout

```text
backend/
├── api.py                    FastAPI and SSE
└── scripts/
    ├── database/             connection, initialization, verification
    ├── generator/            OpenAI client, prompts, answer generation
    ├── professor_fetcher/    SerpAPI faculty lookup
    └── retriever/
        ├── agent/            LangGraph state, nodes, runtime
        ├── sql_search.py
        ├── fulltext_search.py
        ├── hybrid_search.py
        ├── applicant_search.py
        └── recommend.py
```
