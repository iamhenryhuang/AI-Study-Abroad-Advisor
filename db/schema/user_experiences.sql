-- Application experiences that users upload themselves; kept apart from the externally imported applicant_reports.
CREATE TABLE IF NOT EXISTS user_experiences (
    id               BIGSERIAL PRIMARY KEY,
    graduate_school  TEXT NOT NULL,
    country          VARCHAR(100) NOT NULL,
    apply_school     TEXT NOT NULL,
    apply_program    TEXT NOT NULL,
    -- NUMERIC(5,2): a percentage-scale 100 needs 3 integer digits, and (4,2) only goes up to 99.99, which would contradict
    -- the CHECK (gpa <= 100) below -- validation lets the value through but the insert fails with a numeric field overflow.
    gpa              NUMERIC(5,2),
    class_rank       INTEGER,
    class_size       INTEGER,
    experience       JSONB NOT NULL DEFAULT '[]'::jsonb,
    review           TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_user_experiences_gpa
        CHECK (gpa IS NULL OR (gpa >= 0 AND gpa <= 100)),
    CONSTRAINT chk_user_experiences_rank
        CHECK (class_rank IS NULL OR class_rank > 0),
    CONSTRAINT chk_user_experiences_size
        CHECK (class_size IS NULL OR class_size > 0),
    CONSTRAINT chk_user_experiences_rank_size
        CHECK (class_rank IS NULL OR class_size IS NULL OR class_rank <= class_size),
    CONSTRAINT chk_user_experiences_items
        CHECK (jsonb_typeof(experience) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_user_experiences_apply_school_lower
    ON user_experiences (LOWER(apply_school));
CREATE INDEX IF NOT EXISTS idx_user_experiences_school_program_lower
    ON user_experiences (LOWER(apply_school), LOWER(apply_program));
CREATE INDEX IF NOT EXISTS idx_user_experiences_created_at
    ON user_experiences (created_at DESC);

-- Upgrading an existing table: this file uses CREATE TABLE IF NOT EXISTS, so a table that already exists does not get the
-- column definition above; the type fix ((4,2) -> (5,2)) is applied once more here. Idempotent, safe to re-run.
ALTER TABLE user_experiences ALTER COLUMN gpa TYPE NUMERIC(5,2);
