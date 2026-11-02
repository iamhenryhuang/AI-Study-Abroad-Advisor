# db

PostgreSQL + pgvector schema and batch loaders. Runtime connection and operations live in `backend/database/`. Installation is in the [top-level README](../README.md).

## Contents

1. [Tables](#1-tables)
2. [Where data comes from](#2-where-data-comes-from)
3. [Initialization order](#3-initialization-order)
4. [What the crawler writes](#4-what-the-crawler-writes)
5. [Files](#5-files)

## 1. Tables

| Group | Tables | Purpose |
|---|---|---|
| Universities and requirements | `universities`, `programs` | schools, programs, and TOEFL, IELTS, GRE, GPA, fees, recommendation letters, SOP fields |
| One-to-many application data | `program_deadlines`, `program_scholarships`, `program_app_materials` | deadlines, aid, special materials |
| Official text | `web_pages`, `document_chunks` | cleaned pages and retrieval chunks; `fts_vector` is generated automatically, `embedding` can be backfilled |
| Supporting evidence | `program_evidence`, `global_extractions` | content that should not be forced into a column but still has a source (the crawler currently fills only `program_evidence`) |
| Quality control | `review_queue` | fields that failed validation and wait for manual review |
| Faculty | `professors` | name, title, research areas, official link |
| Community | `applicant_reports`, `user_experiences` | GradCafe and 1point3acres reports, and user submissions |

`schema/crawler_tables.sql` is the single source of the crawler tables. `schema/drop_crawler_tables.sql` drops them, and the migration in `data_crawler/db.py` only upgrades additively.

## 2. Where data comes from

| Source | Written by | Main tables |
|---|---|---|
| Official websites | `python -m data_crawler.main ...` | `universities`, the `programs` family, `web_pages`, `document_chunks` |
| Faculty list you provide | `python backend/run.py load-professors` | `professors` |
| GradCafe and 1point3acres | `python db/loaders/load_applicant_reports.py --create-table` | `applicant_reports` |
| User form | FastAPI experiences endpoint | `user_experiences` |

Faculty rows attach through `universities.school_id`, so crawl a school before loading its faculty. A professor with the same name at the same school is upserted on `UNIQUE (university_id, name)`.

`applicant_reports.school_id` has no foreign key, because community cases cover more schools than the crawled `universities`. Query it by its own `school_id` and do not force a join with `universities`.

The community files and the faculty list are not part of this repository; [`data/README.md`](data/README.md) explains how to build them.

## 3. Initialization order

```bash
# Once, at the very beginning
python backend/run.py setup
python backend/run.py init-schema
python backend/run.py init-experience

python -m data_crawler.main --school-id stanford --max-depth 1 --max-pages 10

python backend/run.py load-professors
python backend/run.py verify-db
```

If you do have the community files, `init-full` replaces the first three commands: it rebuilds the crawler schema and also loads the community reports.

What each command does:

- `init-full` rebuilds the crawler schema and loads community application reports.
- `init-schema` drops and recreates the crawler tables.
- `init-experience` creates `user_experiences` idempotently and keeps existing data.
- `clear-crawler-data --yes` clears crawler rows and checkpoints.

After crawling, do not run `init-full` or `init-schema` again: schools, pages, chunks and faculty links would be lost.

## 4. What the crawler writes

The last node of `data_crawler`, `db_writer`:

1. creates or fetches `universities` by `school_id`;
2. upserts `INTERNATIONAL_CS_MASTERS` and its verified structured fields;
3. writes deadlines, scholarships, materials and evidence;
4. stores the official page text in `web_pages`;
5. refreshes `document_chunks` page by page;
6. writes values that failed validation to `review_queue`, never into trusted structured fields.

Without `--dry-run`, and with `DATABASE_URL` set, it writes to the database directly; there is nothing to import from `data_crawler/output`. That folder is for auditing only and is ignored by Git.

## 5. Files

- `schema/crawler_tables.sql`: single source of the crawler schema.
- `schema/drop_crawler_tables.sql`: the DROP step before a rebuild.
- `schema/user_experiences.sql`: the user experience table.
- `schema/applicant_reports.sql`: community report table and full-text index.
- `loaders/load_applicant_reports.py`: cleans and upserts community reports.
- `loaders/load_professors.py`: upserts a faculty list you provide in `data/professors.json`.
- `data/`: empty on purpose. [`data/README.md`](data/README.md) describes the files you build yourself (faculty, community reports, school medians) and their format.
