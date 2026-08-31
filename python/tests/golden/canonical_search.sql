WITH vector_candidates AS (
    SELECT "chunk_id" AS id,
           "embedding" <=> $1::vector AS distance,
           rank() OVER (ORDER BY "embedding" <=> $1::vector) AS rank
    FROM "public"."chunks"
    WHERE "embedding" IS NOT NULL AND "tenant_id" = $2 AND "lang" = ANY($3)
    ORDER BY "embedding" <=> $1::vector
    LIMIT $4
),
text_query AS (
    SELECT (websearch_to_tsquery('english', $5) || websearch_to_tsquery('english', $6)) && !!websearch_to_tsquery('english', $7) AS tsq
),
text_candidates AS (
    SELECT "chunk_id" AS id,
           ts_rank_cd("content_tsv", tsq) AS score,
           rank() OVER (ORDER BY ts_rank_cd("content_tsv", tsq) DESC) AS rank
    FROM "public"."chunks", text_query
    WHERE "content_tsv" @@ tsq AND "tenant_id" = $8 AND "lang" = ANY($9)
    ORDER BY ts_rank_cd("content_tsv", tsq) DESC
    LIMIT $10
),
scored AS (
    SELECT coalesce(v.id, t.id) AS id,
           v.rank AS vector_rank,
           v.distance AS vector_distance,
           coalesce($11 / ($12 + v.rank), 0) AS vector_contribution,
           t.rank AS text_rank,
           t.score AS text_score,
           coalesce($13 / ($12 + t.rank), 0) AS text_contribution
    FROM vector_candidates v
    FULL OUTER JOIN text_candidates t ON v.id = t.id
),
fused AS (
    SELECT id, vector_rank, vector_distance, vector_contribution,
           text_rank, text_score, text_contribution,
           vector_contribution + text_contribution AS fused_score
    FROM scored
)
SELECT f.id,
       (f.fused_score * coalesce(exp(-0.6931471805599453 * greatest(extract(epoch from (now() - "published_at")), 0) / ($14 * 86400.0)), 1.0)) AS score,
       f.fused_score AS fused_score,
       f.vector_rank,
       f.vector_distance,
       f.vector_contribution,
       f.text_rank,
       f.text_score,
       f.text_contribution,
       coalesce(exp(-0.6931471805599453 * greatest(extract(epoch from (now() - "published_at")), 0) / ($14 * 86400.0)), 1.0) AS recency_factor,
       t."content",
       t."title",
       t."url",
       ts_headline('english', t."content", (SELECT tsq FROM text_query), $15) AS highlight
FROM fused f
JOIN "public"."chunks" t ON t."chunk_id" = f.id
ORDER BY score DESC, f.id
LIMIT $16 OFFSET $17
