# Neon removed `pg_search` for new projects. Here is what to do instead.

As of **19 March 2026**, `pg_search` is
[no longer available for new Neon projects](https://neon.com/docs/extensions/pg_search).
Existing projects keep it. New ones do not, which means a project that worked in February
cannot be recreated in April, and a staging branch spun up today will not match production.

The options Neon points at are moving to ParadeDB, or `lakebase_text`. Both are reasonable.
There is a third: **you may not need a BM25 extension at all.**

## What you actually lose

`pg_search` gives you real BM25 ranking. Postgres' built-in full-text search gives you
`ts_rank_cd`, which is a weaker ranking function — it scores by cover density rather than
by term frequency against corpus statistics. On a keyword-only search over a large corpus,
BM25 is meaningfully better and you should not pretend otherwise.

But most people reaching for `pg_search` are not building a keyword-only search engine.
They are building retrieval for a RAG pipeline, where keyword search is one of two signals
and the other is `pgvector`. In that setting the ranking function matters far less than
whether the two signals are combined correctly — and combining them correctly is something
you can do on stock Postgres today.

## The replacement

```bash
pip install pghybrid          # or: npm install pghybrid
```

```bash
pghybrid init --dsn "$DATABASE_URL" --table documents
```

That inspects the table and writes a migration adding a `tsvector` generated column, a GIN
index, and an HNSW index sized for your actual row count. Then:

<!-- check:python -->
```python
from pghybrid import Config, HybridSearch

search = HybridSearch(
    Config(table="documents", text_column="content", vector_column="embedding",
           tsvector_column="fts", paramstyle="pyformat"),
    execute=lambda sql, params: conn.execute(sql, params).fetchall(),
)

results = search.search("renewal notice period", embedding=vec, limit=10)
```

Or copy [`sql/hybrid_search.sql`](../../sql/hybrid_search.sql) and skip the dependency
entirely.

## Rewriting a `pg_search` query

A typical `pg_search` hybrid query looks roughly like this:

```sql
-- before: needs pg_search
select id, paradedb.score(id) as bm25, embedding <=> $1 as distance
from documents
where content @@@ $2
order by bm25 desc
limit 20;
```

The `@@@` operator and `paradedb.score()` both go away. The equivalent shape on stock
Postgres ranks each signal separately and fuses the ranks:

```sql
-- after: pgvector only
with text_query as (
  select (select string_agg(quote_literal(lexeme), ' | ')
          from unnest(to_tsvector('english', $2)))::tsquery as tsq
),
vector_candidates as (
  -- Rank the survivors, not every match: a window in the same select as
  -- order by ... limit stops the index scan from finishing early.
  select id, rank() over (order by distance) as rank
  from (select id, embedding <=> $1 as distance from documents
        where embedding is not null
        order by distance, id limit 50) c
),
text_candidates as (
  select id, rank() over (order by score desc) as rank
  from (select d.id, ts_rank_cd(d.fts, q.tsq) as score
        from documents d, text_query q
        where d.fts @@ q.tsq
        order by score desc, d.id limit 50) c
)
select coalesce(v.id, t.id) as id,
       coalesce(1.0/(60 + v.rank), 0) + coalesce(1.0/(60 + t.rank), 0) as score
from vector_candidates v
full outer join text_candidates t on v.id = t.id
order by score desc limit 20;
```

Three things worth knowing before you paste that:

**Do not sum the raw scores.** The obvious translation —
`0.7 * (1 - distance) + 0.3 * ts_rank_cd(...)` — does not do what it reads as. Cosine
distance is bounded in `[0,1]` and clusters tightly; `ts_rank_cd` is unbounded and small.
The weights set the constants, not the influence. Fusing ranks avoids the problem entirely
because ranks share a scale.

**OR your query terms.** `websearch_to_tsquery('english', 'renewal notice period')` becomes
`'renew' & 'notic' & 'period'` and matches only documents containing all three. Coming from
`@@@`, whose defaults are more forgiving, this reads as "keyword search stopped working" —
and because the vector side still returns rows, the search appears to work while quietly
running on one signal.

**The join must be `full outer`.** An inner join keeps only documents both signals found.

**Rank after limiting, not before.** A `rank()` in the same select as `order by ... limit`
has to see every matching row before the limit can apply, so its cost scales with how many
rows match rather than with the limit — 1.19ms against 0.85ms on 100k rows, widening as the
table grows. Add a tiebreaker to the inner `order by` too: `ts_rank_cd` ties heavily, and
without one the rows chosen at the cut-off can differ between identical runs.

## Is `ts_rank_cd` good enough for you?

An honest answer needs your data, not a benchmark. Measure it:

```bash
pghybrid doctor --dsn "$DATABASE_URL" --table documents
```

That reports recall@k measured against exact search, whether your index is being used, and
where filtered queries are losing rows. If keyword ranking quality turns out to be your
bottleneck rather than fusion or recall, then ParadeDB is the right move and you will have
the evidence for it. If it is not — and for hybrid retrieval it usually is not — you have
one fewer piece of infrastructure to run.

---

Corrections and questions: <https://github.com/pavangupta352/pghybrid/issues>
