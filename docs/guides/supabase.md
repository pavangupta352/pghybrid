# Hybrid search on Supabase, without extensions

Supabase ships `pgvector`, so semantic search is easy. It does not ship `pg_search`, so
BM25 keyword search is not available — ParadeDB has
[no Supabase integration](https://supabase.com/partners/paradedb), and the documented path
is replicating your data into a separate ParadeDB instance, which is a lot of moving parts
for a search box.

You do not need it. Postgres has had full-text search built in since 8.3, and combining it
with vector search well is a matter of getting one query right. This guide is that query.

Everything here runs in the Supabase SQL editor. Nothing needs to be installed.

## 1. The table

If you already have a table with embeddings, skip to step 2. Otherwise:

<!-- check:sql -->
```sql
create extension if not exists vector;

create table documents (
  id         bigserial primary key,
  title      text not null,
  content    text not null,
  created_at timestamptz not null default now(),
  embedding  vector(1536)          -- match your embedding model's dimensions
);
```

## 2. Add the search column and indexes

<!-- check:sql -->
```sql
alter table documents
  add column if not exists fts tsvector
  generated always as (to_tsvector('english', coalesce(content, ''))) stored;

create index if not exists documents_fts_idx
  on documents using gin (fts);

create index if not exists documents_embedding_idx
  on documents using hnsw (embedding vector_cosine_ops);
```

**Use the two-argument `to_tsvector`.** The one-argument form reads
`default_text_search_config`, which makes it `STABLE` rather than `IMMUTABLE`, and Postgres
refuses it in a generated column:

```
ERROR: generation expression is not immutable
```

This is the first thing that goes wrong for most people, and the error message does not
point at the cause.

If your content is not English, change `'english'` to your language, or to `'simple'` for
no stemming — `'simple'` is usually right for product codes, identifiers and mixed-language
corpora.

## 3. The search function

This is the whole thing. Create it once and call it as an RPC.

<!-- check:sql -->
```sql
create or replace function hybrid_search(
  query_text      text,
  query_embedding vector(1536),
  match_count     int  default 10,
  candidates      int  default 50,
  rrf_k           int  default 60
)
returns table (
  id                  bigint,
  title               text,
  content             text,
  score               double precision,
  vector_rank         bigint,
  text_rank           bigint
)
language sql
stable
as $$
  with text_query as (
    -- Postgres' query parsers all combine terms with AND, so a four-word question
    -- usually matches nothing and the keyword half of the search contributes no
    -- ranking at all. OR the terms instead: documents matching more of them still
    -- rank higher, because ts_rank_cd already accounts for that.
    select (
      select string_agg(quote_literal(lexeme), ' | ')
      from unnest(to_tsvector('english', query_text))
    )::tsquery as tsq
  ),
  vector_candidates as (
    -- The window sits outside the limit deliberately. A rank() in the same select as
    -- order by ... limit must see every matching row before the limit applies, so its
    -- cost scales with how many rows match rather than with the limit: 1.19ms against
    -- 0.85ms on 100k rows, widening as the table grows. The inner order by carries a
    -- tiebreaker, without which the rows chosen at the cut-off are arbitrary — and
    -- ts_rank_cd ties heavily.
    select id, distance, rank() over (order by distance) as rank
    from (
      select d.id, d.embedding <=> query_embedding as distance
      from documents d
      where d.embedding is not null
      order by distance, d.id
      limit candidates
    ) c
  ),
  text_candidates as (
    -- Same shape, same two reasons.
    select id, rank() over (order by score desc) as rank
    from (
      select d.id, ts_rank_cd(d.fts, tq.tsq) as score
      from documents d, text_query tq
      where d.fts @@ tq.tsq
      order by score desc, d.id
      limit candidates
    ) c
  )
  select
    d.id,
    d.title,
    d.content,
    coalesce(1.0 / (rrf_k + v.rank), 0.0) + coalesce(1.0 / (rrf_k + t.rank), 0.0) as score,
    v.rank,
    t.rank
  from vector_candidates v
  full outer join text_candidates t on v.id = t.id
  join documents d on d.id = coalesce(v.id, t.id)
  order by score desc, d.id
  limit match_count;
$$;
```

Two details are load-bearing:

**The join is `full outer`.** An inner join would reduce hybrid search to the intersection
of the two result sets, so a document that only one signal found could never compete —
which defeats the purpose.

**Reciprocal Rank Fusion combines ranks, not scores.** The version you will see more often
looks like `0.7 * (1 - (embedding <=> q)) + 0.3 * ts_rank_cd(fts, tsq)`. It reads as
"70% semantic, 30% keyword" and it is not: cosine distance is bounded in `[0,1]` and
clusters tightly, while `ts_rank_cd` is unbounded and typically tiny. The weights describe
the constants, not the influence, and tuning them does not fix it. Ranks share a scale by
construction, so fusing them behaves the way it reads.

## 4. Call it

```ts
const { data, error } = await supabase.rpc("hybrid_search", {
  query_text: "renewal notice period",
  query_embedding: embedding,     // number[] from your embedding model
  match_count: 10,
});
```

From the `pghybrid` package, if you would rather not maintain the SQL yourself:

```ts
import { Config, HybridSearch } from "pghybrid";

const search = new HybridSearch(
  { table: "documents", textColumn: "content", vectorColumn: "embedding", tsvectorColumn: "fts" },
  (sql, params) => pool.query(sql, params).then((r) => r.rows),
);
```

## 5. Row Level Security

The function is `security invoker` by default, so RLS on `documents` applies to the caller
exactly as it does for a normal select. That is what you want: do **not** mark it
`security definer` to "make search work", because it will work for everyone, on every row.

If search is slow for a specific tenant, the cause is usually the filter rather than the
policy — see below.

## 6. When filtered search returns too few rows

Add a tenant filter and you may find a query asking for ten results returns three. The
search is not broken. An approximate index searches a fixed neighbourhood and *then* your
filter removes rows from what it found, so with a selective filter most of the candidates
are discarded.

Two fixes, in order of preference:

```sql
-- pgvector 0.8+: keep scanning until enough rows survive the filter
set local hnsw.iterative_scan = relaxed_order;
set local hnsw.max_scan_tuples = 20000;
```

```sql
-- or raise the candidate pool, which is cruder but works everywhere
select * from hybrid_search('...', $1, 10, 500);
```

Put the filter **inside both** candidate CTEs if you add one — filtering after the fusion
throws away rows that were already ranked, which is the same bug wearing a different hat.

## 7. Check that it is actually working

```bash
pip install pghybrid
pghybrid doctor --dsn "$SUPABASE_DB_URL" --table documents
```

This reports measured recall@10 against exact search, whether your index is being used at
all, and what to change. An under-tuned vector index does not error — it silently returns
worse answers — so the number is worth having.

---

Questions and corrections: <https://github.com/pavangupta352/pghybrid/issues>
