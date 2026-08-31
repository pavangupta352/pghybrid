# pghybrid

**Hybrid search on the Postgres you already have.**

Vector similarity + full-text search, combined by Reciprocal Rank Fusion, on plain
`pgvector`. No `pg_search`, no VectorChord, no Elasticsearch, no vector database, no
extension you need superuser to install.

```bash
pip install pghybrid
```

## Why

Hybrid search on Postgres is a solved problem *if you can install a C extension*.
`pg_search`, VectorChord and `pg_textsearch` are all excellent, and all of them are
extensions — and on managed Postgres you usually cannot install extensions at all.

**`pgvector` is available almost everywhere. BM25 extensions are available almost
nowhere:** not on RDS, Aurora, Cloud SQL, Azure Database, Supabase or Heroku, and
[removed from Neon for new projects in March 2026](https://neon.com/docs/extensions/pg_search).

This is not a claim to search better than ParadeDB — `pg_search`'s BM25 genuinely beats
`ts_rank_cd`, and if you can install it you probably should. It is search you can
actually install.

## The case it exists for

A contract. Someone asks **"renewal notice period"**. The clause that answers it says
*"sixty days written notice prior to the anniversary date"* — it never uses the word
*renewal*. Vector search puts a plausible-but-wrong clause first. Keyword search puts the
clause that uses all three words and answers none of them first. The right answer is
second on both signals and first on neither, which is exactly what rank fusion is for.

## Use it

<!-- check:python -->
```python
from pghybrid import Config, HybridSearch

search = HybridSearch(
    Config(table="chunks", text_column="content", vector_column="embedding",
           tsvector_column="fts", extra_columns=["title"], paramstyle="pyformat"),
    execute=lambda sql, params: conn.execute(sql, params).fetchall(),
)

for row in search.search("renewal notice period", embedding=query_vector, limit=10):
    print(row.score, row.matched_by, row.get("title"))
```

You pass the embedding in. `pghybrid` never calls a model, so it works with OpenAI,
Cohere, Voyage, a local sentence-transformer or anything else, and needs no API key. It
never opens a connection either — it generates SQL and hands it to the driver you already
use. `for_psycopg`, `for_sqlalchemy`, `for_asyncpg` and `for_django` set the placeholder
style for you.

`row.matched_by` tells you which signal found each row — `both`, `vector` or `text` —
which is usually the first thing you want when a search returns the wrong thing.

## Read the statement instead of running it

```python
sql, params = search.build_sql("renewal notice period", query_vector, limit=10)
```

It is one query with a candidate CTE per signal, fused by RRF. Nothing is hidden, and
`pghybrid sql` prints it without a database at all.

## Command line

```bash
pghybrid init   --dsn "$DATABASE_URL" --table chunks   # inspects, writes the migration
pghybrid doctor --dsn "$DATABASE_URL" --table chunks   # measured recall, read-only
```

`doctor` measures recall@k against exact search, sweeps `ef_search`/`probes`, catches
queries that silently fall back to a sequential scan, and reports a `tsvector` that has
stopped matching its text — a failure that returns wrong answers without erroring.

`explain` decomposes a single result set: both ranks, both raw scores, each signal's
contribution, and the near-miss band just below your cut-off.

## Also in TypeScript

[`npm install pghybrid`](https://www.npmjs.com/package/pghybrid) generates byte-identical
SQL, checked on every commit rather than assumed.

---

**Full documentation, the reasoning, and copy-and-paste SQL:
[github.com/pavangupta352/pghybrid](https://github.com/pavangupta352/pghybrid#readme)**

MIT © Pavan Gupta
