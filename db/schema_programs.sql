-- ════════════════════════════════════════════════════════════════
-- programs family -- shared table definitions (the single source of truth)
--
-- This file "only creates tables, never drops them" and is fully idempotent (IF NOT EXISTS), so both sides can use it:
--   db/init_db.sql        DROPs the existing tables, then uses this file -> rebuild style, start from scratch
--   data_crawler/db.py    uses this file directly                         -> additive style, existing data untouched
--
-- These definitions used to be written once in each of the two places above, kept in sync only by a comment, and they drifted easily.
-- Change columns here only, and both sides stay consistent automatically.
-- ════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;

-- ════════════════════════════════════════════════════════════════
-- 1. universities  -- basic school data
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS universities (
    id          SERIAL PRIMARY KEY,
    school_id   VARCHAR(100) UNIQUE NOT NULL,
    name        VARCHAR(255) NOT NULL,
    domain      VARCHAR(255),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ════════════════════════════════════════════════════════════════
-- 2. programs  -- admission targets
-- Currently data_crawler writes one international CS master's target per school; the UNIQUE design still leaves room for future expansion.
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS programs (
    id            SERIAL PRIMARY KEY,
    university_id INTEGER NOT NULL REFERENCES universities(id) ON DELETE CASCADE,

    -- Identification
    degree_type   VARCHAR(20),                  -- MS / PhD / MEng / MPS / MBA
    program_code  TEXT NOT NULL,                -- currently always INTERNATIONAL_CS_MASTERS
    program_name  VARCHAR(200),                 -- currently International Computer Science Master's
    department    VARCHAR(100),                 -- "Computer Science"

    -- Language requirements
    toefl_min          INTEGER,
    toefl_ibt_min      INTEGER,
    toefl_ibt_new_scale_min NUMERIC(2,1),       -- overall on the new TOEFL iBT 1-6 scale, effective 2026-01-21
    toefl_section_requirements TEXT,             -- per-section TOEFL thresholds, merged into one string
    ielts_min          NUMERIC(3,1),
    duolingo_min       INTEGER,
    language_waiver    TEXT,
    english_test_notes TEXT,                     -- English-test validity, format, exceptions and other special rules

    -- GRE requirements
    gre_required       VARCHAR(20),             -- required / optional / not_accepted
    gre_quant_min      INTEGER,
    gre_verbal_min     INTEGER,
    gre_awa_min        NUMERIC(2,1),

    -- GPA requirements
    gpa_min            NUMERIC(3,2),
    gpa_scale          VARCHAR(10),
    gpa_note           TEXT,

    -- Application materials (single-valued fields go here; multi-valued ones go to program_app_materials)
    transcript_copies        INTEGER,
    transcript_format        TEXT,
    rec_letter_count         INTEGER,
    sop_word_limit           INTEGER,
    sop_prompt               TEXT,
    cv_required              BOOLEAN,
    writing_sample_required  BOOLEAN,
    application_fee_usd      INTEGER,
    fee_waiver_available     BOOLEAN,
    fee_waiver_criteria      TEXT,

    -- Tuition
    tuition_per_year_usd  INTEGER,
    tuition_note          TEXT,

    -- How to apply
    application_url       TEXT,
    application_system    TEXT,

    -- Extraction metadata
    source_urls           TEXT[],
    extraction_confidence NUMERIC(3,2),
    last_extracted_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (university_id, program_code)
);

CREATE INDEX IF NOT EXISTS idx_programs_university ON programs(university_id);
CREATE INDEX IF NOT EXISTS idx_programs_degree     ON programs(degree_type);


-- ────────────────────────────────────────────────────────────────
-- updated_at is maintained automatically
--   Child tables used to rely on every writer supplying CURRENT_TIMESTAMP itself, and as soon as one
--   write path forgot, NULL was left behind (the INSERT in load_schools.py did exactly that).
--   A trigger now maintains it in one place, so no writer has to care about this column any more.
-- ────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;


-- ════════════════════════════════════════════════════════════════
-- 3. program_deadlines  -- deadlines (one program can have several)
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS program_deadlines (
    id             SERIAL PRIMARY KEY,
    program_id     INTEGER NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
    deadline_type  VARCHAR(30),                 -- early / regular / international / rolling
    deadline_date  DATE,                        -- legacy compatibility: the same as application_close_date
    application_open_date   DATE,
    application_close_date  DATE,
    decision_release_date   DATE,
    semester       VARCHAR(20),                 -- fall_2026 / spring_2027
    note           TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_deadlines_program ON program_deadlines(program_id);

DROP TRIGGER IF EXISTS program_deadlines_set_updated_at ON program_deadlines;
CREATE TRIGGER program_deadlines_set_updated_at
    BEFORE INSERT OR UPDATE ON program_deadlines
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ════════════════════════════════════════════════════════════════
-- 4. program_scholarships  -- scholarships (one program can have several)
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS program_scholarships (
    id             SERIAL PRIMARY KEY,
    program_id     INTEGER NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
    name           VARCHAR(200),
    amount_usd     INTEGER,
    coverage       VARCHAR(100),                -- full_tuition / partial / stipend_only
    eligibility    TEXT,
    auto_consider  BOOLEAN,
    source_url     TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scholarships_program ON program_scholarships(program_id);

DROP TRIGGER IF EXISTS program_scholarships_set_updated_at ON program_scholarships;
CREATE TRIGGER program_scholarships_set_updated_at
    BEFORE INSERT OR UPDATE ON program_scholarships
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ════════════════════════════════════════════════════════════════
-- 5. program_app_materials  -- multi-valued extras for application materials
--    (the programs table already has the common single-valued fields; this table holds special materials of which there is "more than one")
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS program_app_materials (
    id             SERIAL PRIMARY KEY,
    program_id     INTEGER NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
    material_type  TEXT,                        -- additional_essay / portfolio / video / writing_sample / other
    requirement    TEXT,
    word_limit     INTEGER,
    note           TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_materials_program ON program_app_materials(program_id);

DROP TRIGGER IF EXISTS program_app_materials_set_updated_at ON program_app_materials;
CREATE TRIGGER program_app_materials_set_updated_at
    BEFORE INSERT OR UPDATE ON program_app_materials
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Admission evidence that cannot be safely normalized into a single number/date. The context is kept for later RAG judgment,
-- so important information such as GPA, English thresholds and deadlines is not lost when structured validation fails.
CREATE TABLE IF NOT EXISTS program_evidence (
    id             SERIAL PRIMARY KEY,
    program_id     INTEGER NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
    category       VARCHAR(30) NOT NULL,
    field_name     TEXT NOT NULL DEFAULT 'general',
    evidence_kind  VARCHAR(30) NOT NULL DEFAULT 'context_note',
    evidence_text  TEXT NOT NULL,
    source_excerpt TEXT,
    source_url     TEXT NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (program_id, category, field_name, source_url)
);

CREATE INDEX IF NOT EXISTS idx_program_evidence_program
    ON program_evidence(program_id);
CREATE INDEX IF NOT EXISTS idx_program_evidence_category
    ON program_evidence(category);

DROP TRIGGER IF EXISTS program_evidence_set_updated_at ON program_evidence;
CREATE TRIGGER program_evidence_set_updated_at
    BEFORE INSERT OR UPDATE ON program_evidence
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ════════════════════════════════════════════════════════════════
-- 6. web_pages  -- raw crawled pages
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS web_pages (
    id            SERIAL PRIMARY KEY,
    university_id INTEGER REFERENCES universities(id) ON DELETE CASCADE,
    program_id    INTEGER REFERENCES programs(id) ON DELETE SET NULL,
    url           TEXT UNIQUE NOT NULL,
    passed_types  JSONB NOT NULL,
    raw_text      TEXT NOT NULL,
    char_count    INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_web_pages_university   ON web_pages(university_id);
CREATE INDEX IF NOT EXISTS idx_web_pages_program      ON web_pages(program_id);
CREATE INDEX IF NOT EXISTS idx_web_pages_passed_types ON web_pages USING GIN (passed_types);

-- School-wide / CS-department extraction results: structured data is kept even when no official program has been identified yet
CREATE TABLE IF NOT EXISTS global_extractions (
    id             SERIAL PRIMARY KEY,
    university_id  INTEGER NOT NULL REFERENCES universities(id) ON DELETE CASCADE,
    page_id         INTEGER NOT NULL REFERENCES web_pages(id) ON DELETE CASCADE,
    scope           VARCHAR(30) NOT NULL,
    program_code    TEXT NOT NULL,
    extraction      JSONB NOT NULL,
    confidence      NUMERIC(3,2),
    source_url      TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (page_id, program_code)
);
CREATE INDEX IF NOT EXISTS idx_global_extractions_university ON global_extractions(university_id);

DROP TRIGGER IF EXISTS global_extractions_set_updated_at ON global_extractions;
CREATE TRIGGER global_extractions_set_updated_at
    BEFORE INSERT OR UPDATE ON global_extractions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ════════════════════════════════════════════════════════════════
-- 7. document_chunks  -- full chunk text + optional vectors
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS document_chunks (
    id            SERIAL PRIMARY KEY,
    university_id INTEGER REFERENCES universities(id) ON DELETE CASCADE,
    page_id       INTEGER REFERENCES web_pages(id) ON DELETE CASCADE,
    program_id    INTEGER REFERENCES programs(id) ON DELETE SET NULL,
    school_id     VARCHAR(100) NOT NULL,
    source_url    TEXT NOT NULL,
    passed_types  JSONB NOT NULL,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     vector(1024),
    fts_vector    tsvector,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (page_id, chunk_index)
);

-- HNSW index: m=32 strengthens graph connectivity, ef_construction=200 improves build quality
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 200);

-- FTS index: speeds up keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING GIN (fts_vector);

-- GIN index: multi-type filtering on passed_types
CREATE INDEX IF NOT EXISTS idx_chunks_passed_types ON document_chunks USING GIN (passed_types);

-- Single-column filter indexes
CREATE INDEX IF NOT EXISTS idx_chunks_school  ON document_chunks(school_id);
CREATE INDEX IF NOT EXISTS idx_chunks_pageid  ON document_chunks(page_id);
CREATE INDEX IF NOT EXISTS idx_chunks_program ON document_chunks(program_id);

-- Trigger that updates fts_vector automatically (simple: language-agnostic, so Chinese is not filtered out by stop words / stemming)
DROP TRIGGER IF EXISTS tsvectorupdate ON document_chunks;
CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON document_chunks FOR EACH ROW EXECUTE FUNCTION
tsvector_update_trigger(fts_vector, 'pg_catalog.simple', chunk_text);


-- ════════════════════════════════════════════════════════════════
-- 8. review_queue  -- manual-review queue for fields that failed data_crawler validation
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS review_queue (
    id             SERIAL PRIMARY KEY,
    university_id  INTEGER REFERENCES universities(id) ON DELETE CASCADE,
    page_id        INTEGER REFERENCES web_pages(id) ON DELETE CASCADE,
    program_code   TEXT,
    field_name     VARCHAR(100),
    field_value    TEXT,
    reason         VARCHAR(50),                 -- hallucination_detected etc.
    source_excerpt TEXT,
    status         VARCHAR(20) DEFAULT 'pending',  -- pending / resolved / rejected
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_review_queue_university ON review_queue(university_id);
CREATE INDEX IF NOT EXISTS idx_review_queue_status     ON review_queue(status);


-- ════════════════════════════════════════════════════════════════
-- 9. professors  -- seed data for the faculty list (written by hand through db/load_professors.py,
--    not produced by data_crawler; data_crawler does not process faculty pages for now)
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS professors (
    id             SERIAL PRIMARY KEY,
    university_id  INTEGER NOT NULL REFERENCES universities(id) ON DELETE CASCADE,
    name           VARCHAR(200) NOT NULL,
    title          VARCHAR(100),                -- Professor / Associate Professor / Assistant Professor
    research_areas TEXT[],
    profile_url    TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (university_id, name)
);

CREATE INDEX IF NOT EXISTS idx_professors_university ON professors(university_id);
