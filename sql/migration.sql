-- pghybrid, schema changes for hybrid search
-- https://github.com/pavangupta352/pghybrid
--
-- Run this once. It adds a searchable tsvector column and the two indexes the
-- query needs. Nothing here requires superuser or an extension beyond pgvector.
--
-- Adjust the identifiers marked CHANGE ME to match your table.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 1. The tsvector column.
--
-- The two-argument form of to_tsvector is required and the one-argument form
-- will not work. to_tsvector(text) reads default_text_search_config, which
-- makes it STABLE rather than IMMUTABLE, and Postgres refuses STABLE
-- expressions in a generated column:
--
--     ERROR: generation expression is not immutable
--
-- Naming the configuration explicitly also means the column keeps its meaning
-- if the database default is ever changed underneath you.
--
-- 'english' applies English stemming, which is wrong for most other languages.
-- Use your own language ('french', 'german', 'spanish', ...) or 'simple' for
-- no stemming at all, 'simple' is usually the right choice for identifiers,
-- product codes and mixed-language corpora.
-- ---------------------------------------------------------------------------
ALTER TABLE chunks                                 -- CHANGE ME: your table
    ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content, ''))   -- CHANGE ME: config, column
    ) STORED;

-- ---------------------------------------------------------------------------
-- 2. The full-text index.
--
-- GIN is the right choice for a column that is searched far more often than it
-- is written. GiST builds faster and stays smaller but answers more slowly,
-- which is the wrong trade for a search index.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS chunks_fts_idx          -- CHANGE ME: your table
    ON chunks USING gin (fts);

-- ---------------------------------------------------------------------------
-- 3. The vector index.
--
-- HNSW is the default worth reaching for: it gives better recall than IVFFlat
-- at the same latency, and it does not need the table to be populated before
-- you build it. m = 16 and ef_construction = 64 are pgvector's own defaults,
-- stated here so they are visible rather than implied. Raising
-- ef_construction improves recall and costs build time; raising m improves
-- recall on high-dimensional data and costs memory.
--
-- The operator class must match the distance operator used at query time:
--     vector_cosine_ops  <=>   cosine distance          (most embeddings)
--     vector_l2_ops      <->   Euclidean distance
--     vector_ip_ops      <#>   negative inner product
-- A mismatch does not error. The planner simply ignores the index and the
-- query falls back to a sequential scan, which looks like "vector search got
-- slow" rather than like a mistake.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS chunks_embedding_idx    -- CHANGE ME: your table
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- On a large, live table, build indexes without holding a write lock. This
-- cannot run inside a transaction block, so it is left commented out:
--
--     CREATE INDEX CONCURRENTLY chunks_embedding_idx
--         ON chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Optional: halve the index with halfvec.
--
-- Half-precision floats hold about three decimal digits, which is more than an
-- embedding's meaningful precision. The index becomes roughly half the size and
-- builds in roughly half the time, usually with no measurable recall loss. The
-- stored column stays full precision; only the index is reduced.
--
-- The query must then cast its argument the same way:
--     ORDER BY embedding::halfvec(1536) <=> $1::halfvec(1536)
--
--     CREATE INDEX chunks_embedding_half_idx ON chunks
--         USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops);
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Optional: filtered search that keeps its recall.
--
-- An approximate index searches a fixed neighbourhood, then your WHERE clause
-- removes rows from what it found. With a selective filter, one tenant out of
-- ten thousand, almost everything the index returned is discarded and you get
-- three results when you asked for ten. The search is not broken; it never
-- looked in the right place.
--
-- pgvector 0.8.0 and later fix this properly: the index keeps scanning until it
-- has enough rows that survive the filter. Set this per session or per
-- transaction, not globally.
--
--     SET hnsw.iterative_scan = relaxed_order;   -- or strict_order
--     SET hnsw.max_scan_tuples = 20000;          -- bounds the worst case
--
-- On older pgvector, a partial index per tenant is the usual workaround.
-- ---------------------------------------------------------------------------

ANALYZE chunks;                                    -- CHANGE ME: your table
