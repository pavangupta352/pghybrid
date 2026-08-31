-- pghybrid — hybrid search on plain Postgres
-- https://github.com/pavangupta352/pghybrid
--
-- Vector similarity + full-text search, combined by Reciprocal Rank Fusion.
-- Requires pgvector. Requires nothing else — no pg_search, no VectorChord, no
-- Elasticsearch, no vector database. It runs on RDS, Aurora, Cloud SQL, Azure,
-- Supabase, Neon, Heroku and anything self-hosted, unchanged.
--
-- Run sql/migration.sql first if you have not already.
--
-- Parameters
--   $1  the query embedding, e.g. '[0.1,0.2,...]'::vector
--   $2  the query text, e.g. 'renewal notice period'
--   $3  candidates per signal (50 is a good default)
--   $4  rows to return
--
-- Adjust the identifiers marked CHANGE ME to match your table.

WITH
-- ---------------------------------------------------------------------------
-- The keyword query.
--
-- Every Postgres query parser combines terms with AND: websearch_to_tsquery
-- turns 'renewal notice period' into 'renew' & 'notic' & 'period', which
-- matches only documents containing all three words. As a filter that is
-- correct. As the keyword half of a hybrid search it is quietly destructive —
-- a four or five word question usually matches nothing at all, the keyword
-- signal contributes no ranking, and the search silently degrades to
-- vector-only without reporting that anything went wrong.
--
-- So the terms are OR-ed instead. Documents matching more of them still rank
-- higher, because ts_rank_cd already accounts for that: precision comes back
-- through ranking rather than through exclusion. Feeding the text through
-- to_tsvector first means stemming and stop-word removal are Postgres' own.
--
-- Swap in websearch_to_tsquery('english', $2) if you genuinely want AND, or if
-- you need quoted phrases.
--
-- If you swap it in for -negation, read this first. websearch reads a leading
-- dash as NOT, so the excluded rows leave the text candidates — and nothing
-- removes them from the vector candidates below. They come back with a vector
-- rank and no text rank, and RRF pays the best vector hit 1/(k+1), the largest
-- single contribution available, so the row you excluded can rank first. To
-- exclude properly, add the same predicate to BOTH candidate CTEs:
--
--     AND NOT coalesce(fts @@ websearch_to_tsquery('english', $3), false)
--
-- inside each subquery, before its LIMIT. (The coalesce is not decoration: a
-- NULL tsvector would otherwise make the whole predicate NULL and drop the row.)
-- ---------------------------------------------------------------------------
-- A query of nothing but stop words yields NULL here, which matches no rows and
-- contributes no text candidates. That is the wanted behaviour: the vector side
-- still answers, rather than the whole query failing.
text_query AS (
    SELECT (
        SELECT string_agg(quote_literal(lexeme), ' | ')
        FROM unnest(to_tsvector('english', $2))       -- CHANGE ME: text search config
    )::tsquery AS tsq
),

-- ---------------------------------------------------------------------------
-- One candidate set per signal.
--
-- Note where the WHERE clause lives. Any filter you need (tenant_id, a date
-- range, a document type) belongs INSIDE both of these CTEs, so that both
-- signals rank the same subset of rows. Filtering after the fusion instead is
-- the single most common way a hand-rolled implementation goes wrong: it
-- throws away rows that were already ranked, so a query returns four results
-- when you asked for ten and the shortfall looks like bad recall.
-- ---------------------------------------------------------------------------
-- The window sits outside the LIMIT, and that is not cosmetic. A rank() in the same
-- SELECT as ORDER BY ... LIMIT has to see every matching row before the limit can
-- apply, so its cost scales with how many rows match rather than with the limit.
-- Ranking the rows that survive is the same answer for less work: 1.19ms against
-- 0.85ms on 100k rows, and the gap widens as the table grows.
--
-- The inner ORDER BY carries a tiebreaker too. Without one the rows chosen at the
-- cut-off are arbitrary, and ties are not rare: on a benchmark corpus ts_rank_cd
-- produced just 3 distinct values across 3,399 matching rows, so which candidates came
-- back could change between identical runs.
vector_candidates AS (
    SELECT id, distance, rank() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT
            id,
            embedding <=> $1::vector AS distance
        FROM chunks                                -- CHANGE ME: your table
        WHERE embedding IS NOT NULL
          -- AND tenant_id = $5                    -- filters go here, not later
        ORDER BY distance, id
        LIMIT $3
    ) candidates
),

-- Same shape, same two reasons. See the comment above vector_candidates.
text_candidates AS (
    SELECT id, score, rank() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT
            id,
            ts_rank_cd(fts, tsq) AS score
        FROM chunks, text_query                    -- CHANGE ME: your table
        WHERE fts @@ tsq
          -- AND tenant_id = $5                    -- the same filters, repeated
        ORDER BY score DESC, id
        LIMIT $3
    ) candidates
),

-- ---------------------------------------------------------------------------
-- Reciprocal Rank Fusion.
--
--     score = Σ  weight / (k + rank)
--
-- RRF combines RANKS, not scores, and that is the whole point. The obvious
-- alternative — 0.7 * (1 - cosine_distance) + 0.3 * ts_rank_cd — reads as
-- "70% semantic, 30% keyword" and is not. Cosine distance is bounded in [0,1]
-- and clusters tightly on real embeddings: the top fifty candidates might all
-- sit between 0.62 and 0.81, a span of 0.19. ts_rank_cd is unbounded and on
-- short chunks typically lands near 0.02. The weights you wrote describe the
-- constants, not the influence, and no amount of tuning fixes it because the
-- spans are set by the scoring functions rather than by you.
--
-- Ranks share a scale by construction, so under RRF the weights mean what they
-- say. k = 60 is from Cormack, Clarke & Buettcher (2009); it flattens the gap
-- between the top few ranks so neither signal wins on its first result alone.
--
-- The join must be FULL OUTER. An INNER JOIN here reduces hybrid search to the
-- intersection of the two result sets, which is a different and much worse
-- product: a document that only one signal found can no longer compete.
-- ---------------------------------------------------------------------------
scored AS (
    SELECT
        COALESCE(v.id, t.id)                        AS id,
        v.rank                                      AS vector_rank,
        v.distance                                  AS vector_distance,
        t.rank                                      AS text_rank,
        t.score                                     AS text_score,
        COALESCE(1.0 / (60 + v.rank), 0)            AS vector_contribution,
        COALESCE(1.0 / (60 + t.rank), 0)            AS text_contribution
    FROM vector_candidates v
    FULL OUTER JOIN text_candidates t ON v.id = t.id
),

fused AS (
    SELECT *, vector_contribution + text_contribution AS fused_score
    FROM scored
)

-- ---------------------------------------------------------------------------
-- The final page.
--
-- ts_headline is evaluated here and nowhere else. It re-parses the document
-- text, so running it inside the candidate CTEs would pay that cost for every
-- candidate instead of for the handful of rows actually being returned.
--
-- To add recency decay, multiply fused_score by
--     exp(-ln(2) * extract(epoch from (now() - created_at)) / (half_life_days * 86400))
-- which halves a row's score every half_life_days. COALESCE it to 1.0 so rows
-- with no timestamp are left alone rather than erased.
-- ---------------------------------------------------------------------------
SELECT
    f.id,
    f.fused_score AS score,
    f.vector_rank,
    f.vector_distance,
    f.text_rank,
    f.text_score,
    f.vector_contribution,
    f.text_contribution,
    c.title,                                       -- CHANGE ME: your columns
    c.content,
    ts_headline('english', c.content, (SELECT tsq FROM text_query),
                'StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=8, MaxWords=30')
        AS highlight
FROM fused f
JOIN chunks c ON c.id = f.id                       -- CHANGE ME: your table
ORDER BY score DESC, f.id
LIMIT $4;
