-- ════════════════════════════════════════════════════════════════
-- Study Abroad RAG — the schema follows what data_crawler can extract/write
-- Real data source: written by data_crawler (the LangGraph crawler)
-- db/data/schools_data.json is only fake test data that matches this schema
-- ════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the old tables first (mind the order: drop the referenced ones last)
-- professors must be listed explicitly: it hangs off universities through a university_id foreign key,
-- and if it is left out, old rows become orphans pointing at ids that no longer exist once universities is rebuilt,
-- so the professor-list query finds nothing through its JOIN.
DROP TABLE IF EXISTS review_queue            CASCADE;
DROP TABLE IF EXISTS document_chunks         CASCADE;
DROP TABLE IF EXISTS web_pages               CASCADE;
DROP TABLE IF EXISTS global_extractions       CASCADE;
DROP TABLE IF EXISTS program_evidence        CASCADE;
DROP TABLE IF EXISTS program_app_materials   CASCADE;
DROP TABLE IF EXISTS program_scholarships    CASCADE;
DROP TABLE IF EXISTS program_deadlines       CASCADE;
DROP TABLE IF EXISTS professors              CASCADE;
DROP TABLE IF EXISTS programs                CASCADE;
DROP TABLE IF EXISTS universities            CASCADE;

-- The table definitions live in db/schema_programs.sql (the single source of truth, shared with data_crawler).
-- This file only "clears first"; run.py init-schema then runs that shared file to create the tables.
-- If you run this file by hand with psql, remember to follow it with: \i db/schema_programs.sql
