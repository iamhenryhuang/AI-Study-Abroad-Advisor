<div align="center">

# A Source-Aware Agentic RAG Architecture for Study-Abroad Admissions Advising: Design and Evaluation

**TAAI 2026 · Domestic Track · (poster presentation)**

[Po-Yuan Huang](https://github.com/iamhenryhuang)<sup>1</sup>, [Ting-Han Wu](https://github.com/112703028)<sup>1</sup>, [Kai-Ting Chen](https://github.com/ckt1022)<sup>1</sup> (equal contribution)

<sup>1</sup>Department of Computer Science, National Chengchi University

Paper (coming soon) · [Contents](#contents) · [Quick start](#quick-start) · [Citation](#citation)

[![TAAI 2026](https://img.shields.io/badge/TAAI-2026-4b44ce.svg)](#citation)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-green.svg)](LICENSE)

</div>

**TL;DR.** A source-aware agentic RAG that routes study-abroad questions by source authority, not just relevance, and abstains when official evidence is missing.

**Keywords:** source-aware agentic RAG, study-abroad advising, text-to-SQL, hybrid retrieval, evidence verification

Large language models answer study-abroad questions with confidence and often get them wrong: deadlines, language requirements, outdated faculty information, and forum rumors stated as official rules. The sources behind these questions differ in authority (official requirements, community-reported experiences, faculty research), so relevance alone should not decide what may answer a question. This repository routes each question by the kind of source it needs, labels community experiences as unofficial, and abstains when official evidence is missing instead of guessing.

## How it works

| Question about | Handled by | Treated as |
|---|---|---|
| Official requirements (deadline, TOEFL/IELTS, GRE, fees) | Text-to-SQL over structured records extracted from official pages | Official |
| Admission experiences | Retrieval over community reports | Unofficial, labeled as such |
| Faculty research | Query-time search (SerpAPI) over faculty profiles | Current, not stored |

Before an answer is generated, a bounded evidence check (the Verifier) decides whether the retrieved evidence covers what was asked. If not, the system retries once with a full-text search over the crawled pages, then rewrites the query and retries once more, and otherwise abstains. A Critic checks the generated answer afterwards and appends a warning to statements that look unsupported.

## Contents

| Directory | Contribution | Released |
|---|---|---|
| [`backend/`](backend/) | FastAPI service and the LangGraph agent: decomposition and routing, text-to-SQL, full-text and hybrid retrieval, the Verifier, answer generation and the Critic | source, CLI, API ([README](backend/README.md)) |
| [`data_crawler/`](data_crawler/) | Ingestion pipeline: crawl official pages, extract structured fields with the source excerpt, validate each excerpt against the page, chunk and write to the database | source, CLI, per-school root URLs ([README](data_crawler/README.md)) |
| [`db/`](db/) | PostgreSQL + pgvector schema, migrations and loaders | source, SQL, lists ([README](db/README.md)) |

## Repository layout

```text
backend/            FastAPI service, command line and the agent
  api.py, run.py    the two entry points
  retriever/        agent/ (LangGraph), search/ (one module per retrieval path), recommendation
  generator/        OpenAI client, answer prompts, answer generation
  professor_fetcher/ query-time faculty lookup
  database/         connection, schema commands, user experiences
data_crawler/       ingestion graph: nodes/ (school and page steps), config/ (schools, parameters)
db/                 schema/ (SQL), loaders/ (batch loaders), data/ (files you provide)
```

## News

- **2026-10** The paper is accepted to **TAAI 2026** (Domestic Track) and will appear in the official proceedings. It is assigned a poster presentation.

## Installation

```bash
git clone <this-repository-url>
cd <this-repository>
```

| Component | Install | Hardware |
|---|---|---|
| backend and data_crawler | `pip install -r requirements.txt`, then `playwright install chromium` | CPU is enough; the language model is called through the OpenAI API |
| database | `docker compose up -d db` (PostgreSQL + pgvector, host port 5434) | disk space for the Docker image |
| embeddings (optional) | BGE-M3 and a BGE reranker through `sentence-transformers`, then `python -m data_crawler.backfill_embeddings` | a GPU helps; until vectors exist the system uses PostgreSQL full-text search and loads no model |

## Data

This repository contains code only. It does not ship any data, and in particular it does not redistribute third-party content:

- Crawled text from university websites.
- GradCafe and 1point3acres records, and statistics derived from them.
- The faculty list.

How each kind of data is built:

- **School information is crawled.** Run `data_crawler` against the schools you want, and mind each site's terms of use.
- **Community reports and the faculty list are collected by hand.** We built some of them manually, partly because the sites block automated access and partly because it was simpler, and then loaded them from JSON files. The system does not depend on how the files were made, only on their format. `db/loaders/load_applicant_reports.py` reads `db/data/gradcafe_results.json` and `db/data/1point3.json`, and `load-professors` reads `db/data/professors.json`. Without the community files the Experience path has nothing to retrieve.
- **School recommendation:** `backend/retriever/recommend.py` reads `db/data/1point3_distribution.json`, which you provide in the same way.

[`db/data/README.md`](db/data/README.md) lists every file the system expects there and its format, including the faculty list that `load-professors` reads.

## Models used

| Role | Model | Configured by |
|---|---|---|
| Planning, verification, critique, refinement, text-to-SQL | `gpt-4o-mini` | `OPENAI_MODEL` |
| Final answer generation | `gpt-4o` | `OPENAI_ANSWER_MODEL` |
| Embeddings (optional, hybrid retrieval) | `BAAI/bge-m3` | `BGE_EMBED_MODEL_PATH` |
| Reranker (optional) | `BAAI/bge-reranker-v2-m3` | `BGE_RERANKER_MODEL_PATH` |
| Faculty lookup | SerpAPI | `SERPAPI_KEY` |

`.env.example` sets `OPENAI_MODEL=gpt-4.1` by default; set it to `gpt-4o-mini` to match the paper. The OpenAI base URL can point to any compatible endpoint through `OPENAI_BASE_URL`.

## Quick start

```bash
# 1. Database
docker compose up -d db

# 2. Python environment
python -m venv .venv
source .venv/Scripts/activate      # Git Bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
playwright install chromium

# 3. Configuration (fill in OPENAI_API_KEY and SERPAPI_KEY)
cp .env.example backend/.env

# 4. Empty schema
python backend/run.py setup
python backend/run.py init-schema
python backend/run.py init-experience

# 5. Crawl a school's official pages, load faculty (optional; build db/data/professors.json first), check the result
python -m data_crawler.main --school-id stanford --max-depth 1 --max-pages 10
python backend/run.py load-professors
python backend/run.py verify-db
```

Ask the agent from the command line, or start the API:

```bash
python backend/run.py agent "Stanford CS MS 申請費是多少？"
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints: `POST /api/chat` (SSE agent), `GET /api/health`, `POST` and `GET /api/experiences`.

The API checks the database when it starts and exits with an error if it cannot connect (the wait is 5 seconds, set `DB_CONNECT_TIMEOUT` to change it), so start the database first.

The output is research software, not official admissions advice. Check requirements on the school's website before you apply.

## FAQ

<details>
<summary>The agent says it has no data for a school.</summary>

The database is empty until you crawl. Run `data_crawler.main --school-id <id>` for that school (the ids are in `data_crawler/config/root_url.py`). Questions about a school with no official data are meant to be answered with "not indexed", not with a guess.

</details>

<details>
<summary>The Experience path returns nothing.</summary>

Community reports are not included in this repository; see [Data](#data). Without `db/data/gradcafe_results.json` and `db/data/1point3.json` there is nothing for it to retrieve.

</details>

<details>
<summary>Do I need a GPU or the embedding models?</summary>

No. Hybrid retrieval is optional. Until you run `python -m data_crawler.backfill_embeddings`, or when the embedding model is missing, or with `ENABLE_HYBRID_SEARCH=false`, the system uses PostgreSQL full-text search.

</details>

<details>
<summary>I cannot connect to the database.</summary>

`docker-compose.yml` publishes the database on host port **5434**, not 5432, so `DATABASE_URL` must use `127.0.0.1:5434`. On Windows with Docker Desktop, use `127.0.0.1` rather than `localhost`.

</details>

<details>
<summary>Why is there still Chinese in the code?</summary>

Code comments, docstrings and log messages are in English. The remaining Chinese is deliberate because it is part of the behavior, not documentation: the generator and agent prompts and the labels sent to the LLM, the answer text shown to users (the system answers in Traditional Chinese), and the keyword, alias and regex data that match Chinese questions and forum posts (for example school names such as 史丹佛, or the `1point3acres` labels). Translating those strings would change how the agent behaves.

</details>

<details>
<summary>I ran <code>init-schema</code> again and my crawled data is gone.</summary>

`init-schema` drops and recreates the crawler tables. Run it once before crawling, and do not run it again afterwards.

</details>

## Citation

```bibtex
@inproceedings{huang2026sourceaware,
  title     = {A Source-Aware Agentic RAG Architecture for Study-Abroad Admissions Advising: Design and Evaluation},
  author    = {Huang, Po-Yuan and Wu, Ting-Han and Chen, Kai-Ting},
  booktitle = {Proceedings of the 31st International Conference on Technologies and Applications of Artificial Intelligence (TAAI)},
  year      = {2026},
  note      = {Domestic Track}
}
```

## Acknowledgements

This release builds on [LangGraph](https://github.com/langchain-ai/langgraph), [FastAPI](https://fastapi.tiangolo.com/), [PostgreSQL](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector), [Playwright](https://playwright.dev/), [trafilatura](https://github.com/adbar/trafilatura) and [BGE](https://huggingface.co/BAAI/bge-m3).

## Contact

Questions and bug reports: please open an issue in this repository.

## Copyright and Terms of Use

Copyright (c) 2026 the authors.

- **Code**: Apache License 2.0. See [LICENSE](LICENSE).
- Third-party web content, GradCafe and 1point3acres records are **not** redistributed here. The license covers the code in this repository; it does not cover content you crawl with it, which remains subject to the terms of its source.
