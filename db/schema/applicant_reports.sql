-- ════════════════════════════════════════════════════════════════
-- applicant_reports -- application-result reports from online communities (experience data, unofficial)
--
-- Data sources:
--   gradcafe  -- GradCafe (English, structured, 995 rows)
--   1point3   -- the 1point3acres forum (Simplified Chinese, semi-structured, 596 rows)
--
-- Nature: these are individual applicants' "admitted / rejected" reports -- subjective, biased samples, not authoritative.
--       They are very different from programs (the official application requirements): they answer "what do admitted applicants' backgrounds look like",
--       not "what scores does the school require". When RAG cites them it must label them as unofficial experience,
--       and never present them as an admission threshold or a guarantee (the guardrail is in the backend generation prompt).
--
-- Idempotent: CREATE IF NOT EXISTS. It is not one of the crawler tables, so run.py init-schema does not drop it.
-- ════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS applicant_reports (
    id            SERIAL PRIMARY KEY,
    source        VARCHAR(20) NOT NULL,        -- 'gradcafe' / '1point3'
    source_url    TEXT,

    -- School: the raw school name is always kept; school_id maps onto the universities alias system where possible, and stays NULL when nothing matches
    school_raw    TEXT NOT NULL,
    school_id     VARCHAR(100),                -- maps to SCHOOL_ALIASES (mit/cmu...), NULL when nothing matches

    program       TEXT,
    degree_level  VARCHAR(20),                 -- masters / phd / mfa / other...

    -- Result: the raw values of the two sources are normalized into one common set
    decision      VARCHAR(20),                 -- accepted / rejected / waitlisted / interview
                                               -- / offer / ad_no_fund / ad_small_fund / other

    -- GPA: a value that can be parsed onto a 4.0 scale goes into gpa (for statistics/filtering); the raw string is always kept in gpa_raw (to avoid cleaning distortion)
    gpa           NUMERIC(4,2),
    gpa_raw       TEXT,

    season        VARCHAR(30),                 -- 'Fall 2026' etc. (the raw string is kept)
    notes         TEXT,                        -- gradcafe.notes / the short content of 1point3

    fts_vector    tsvector,                    -- for full-text search (so the RAG fallback can find it)
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (source, source_url)
);

CREATE INDEX IF NOT EXISTS idx_applicant_reports_school   ON applicant_reports(school_id);
CREATE INDEX IF NOT EXISTS idx_applicant_reports_source   ON applicant_reports(source);
CREATE INDEX IF NOT EXISTS idx_applicant_reports_decision ON applicant_reports(decision);
CREATE INDEX IF NOT EXISTS idx_applicant_reports_fts      ON applicant_reports USING GIN (fts_vector);

-- fts_vector is maintained automatically (simple config: language-agnostic, so Chinese is not damaged by English stemming; same as document_chunks)
-- It indexes school_raw + program + notes, so queries such as "CMU CS admitted" hit
CREATE OR REPLACE FUNCTION applicant_reports_fts_trigger() RETURNS trigger AS $$
BEGIN
    NEW.fts_vector :=
        to_tsvector('pg_catalog.simple',
            coalesce(NEW.school_raw, '') || ' ' ||
            coalesce(NEW.program, '')    || ' ' ||
            coalesce(NEW.notes, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS applicant_reports_fts_update ON applicant_reports;
CREATE TRIGGER applicant_reports_fts_update
    BEFORE INSERT OR UPDATE ON applicant_reports
    FOR EACH ROW EXECUTE FUNCTION applicant_reports_fts_trigger();
