-- Drops the crawler tables. `python backend/run.py init-schema` runs this file and then crawler_tables.sql.
-- This deletes all crawled data; the table definitions live in crawler_tables.sql.
-- If you run this file by hand with psql, remember to follow it with: \i db/schema/crawler_tables.sql

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
