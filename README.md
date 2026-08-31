# pghybrid

**Hybrid search on the Postgres you already have.**

Vector similarity + full-text search, combined by Reciprocal Rank Fusion, on plain
`pgvector`. No `pg_search`, no VectorChord, no Elasticsearch, no vector database, no
extension you need superuser to install.

[![PyPI](https://img.shields.io/pypi/v/pghybrid?color=306998&label=pypi)](https://pypi.org/project/pghybrid/)
[![npm](https://img.shields.io/npm/v/pghybrid?color=cb3837&label=npm)](https://www.npmjs.com/package/pghybrid)
[![CI](https://github.com/pavangupta352/pghybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/pavangupta352/pghybrid/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The problem, in one table

A contract. Someone asks **"renewal notice period"**. The clause that answers it says
*"sixty days written notice prior to the anniversary date"* — it never uses the word
*renewal*.

| # | vector only | keyword only | **pghybrid** |
|---|---|---|---|
| 1 | Automatic extension | Renewal pricing | ✅ **Termination for convenience** |
| 2 | ✅ Termination for convenience | ✅ Termination for convenience | Renewal pricing |
| 3 | Subscription term | Renewal terms | Renewal terms |
| 4 | Fees and invoicing | Notice requirements | Notice requirements |
| 5 | Service levels | — | Automatic extension |

Semantic search puts a plausible-but-wrong clause first. Keyword search puts the clause
that *uses all three words and answers none of them* first. The right answer is second on
both, and first on neither — which is exactly the case rank fusion exists to fix.

Reproduce it yourself in about a minute: [`examples/`](examples/).

## Why this exists

Hybrid search on Postgres is a solved problem *if you can install a C extension*.
`pg_search`, VectorChord and `pg_textsearch` are all excellent, and all of them are
extensions. On managed Postgres you usually cannot install extensions at all.

**`pgvector` is available almost everywhere. BM25 extensions are available almost nowhere.**

| | pgvector | `pg_search` (ParadeDB) | **pghybrid** |
|---|---|---|---|
| Amazon RDS | ✅ | ❌ not in the extension allowlist | ✅ |
| Amazon Aurora | ✅ | ❌ | ✅ |
| Google Cloud SQL | ✅ | ❌ | ✅ |
| Azure Database | ✅ | ❌ | ✅ |
| Supabase | ✅ built in | ❌ [no integration](https://supabase.com/partners/paradedb) — needs replication to a separate instance | ✅ |
| Neon | ✅ | ❌ [removed for new projects, March 2026](https://neon.com/docs/extensions/pg_search) | ✅ |
| Heroku Postgres | ✅ | ❌ | ✅ |
| Self-hosted | ✅ | ✅ | ✅ |

This is the whole pitch. It is **not** *"better search than ParadeDB"* — `pg_search`'s BM25
is genuinely better than `ts_rank_cd`, and if you can install it, you probably should.
This is *"search you can actually install."*

## Install

```bash
pip install pghybrid          # Python
npm install pghybrid          # TypeScript / JavaScript
```

Zero runtime dependencies in both. `pghybrid` generates SQL and hands it to the driver you
already use — psycopg, asyncpg, SQLAlchemy, node-postgres, Drizzle, Supabase. It never
opens a connection of its own and never calls an embedding provider.

Or **install nothing at all**: [`sql/hybrid_search.sql`](sql/hybrid_search.sql) and
[`sql/migration.sql`](sql/migration.sql) are complete, commented and standalone. Paste them
into the Supabase SQL editor and you are done. Copying them instead of installing the
package is a supported way to use this project.

## 30-second quickstart

```bash
pghybrid init --dsn $DATABASE_URL --table chunks     # inspects your table, writes the migration
```

```python
from pghybrid import Config, HybridSearch

search = HybridSearch(
    Config(table="chunks", text_column="content", vector_column="embedding",
           tsvector_column="fts", paramstyle="pyformat"),
    execute=lambda sql, params: conn.execute(sql, params).fetchall(),
)

for row in search.search("renewal notice period", embedding=query_vector, limit=10):
    print(row.score, row.get("title"))
```

```ts
import { Config, HybridSearch } from "pghybrid";

const search = new HybridSearch(
  { table: "chunks", textColumn: "content", vectorColumn: "embedding", tsvectorColumn: "fts" },
  (sql, params) => pool.query(sql, params).then((r) => r.rows),
);

const rows = await search.search("renewal notice period", { embedding, limit: 10 });
```

You pass the embedding in. `pghybrid` never calls a model, so it works with OpenAI, Cohere,
Voyage, a local sentence-transformer, or anything else — and it needs no API key.

## Two decisions that make it work

Most hand-rolled Postgres hybrid search is subtly broken in the same two ways.

### 1. Your weights are lying to you

This is the shape almost everyone writes:

```sql
0.7 * (1 - (embedding <=> $1)) + 0.3 * ts_rank_cd(fts, query)
```

It reads as *70% semantic, 30% keyword*. It is not. Cosine distance is bounded in `[0,1]`
and clusters tightly — your top 50 candidates might all sit between 0.62 and 0.81, a span
of 0.19. `ts_rank_cd` is **unbounded** and on short chunks typically lands near 0.02.

The weights describe the constants, not the influence. **Tuning them cannot fix it**,
because the spans are set by the scoring functions, and `ts_rank_cd`'s span shifts with
document length and corpus statistics.

`pghybrid explain` measures this on your own data:

```
weighted fusion          nominal    span     effective
  vector                   70.0%    0.190       85.2%
  text                     30.0%    0.077       14.8%

rrf fusion (default)     nominal    span     effective
  vector                   50.0%    0.016       50.1%
  text                     50.0%    0.016       49.9%
```

Reciprocal Rank Fusion combines **ranks**, which share a scale by construction, so the
weights mean what they say:

```
score = Σ  weight / (k + rank)
```

`k = 60` is from [Cormack, Clarke & Buettcher (2009)](https://dl.acm.org/doi/10.1145/1571941.1572114).

### 2. Your keyword search matches nothing

Every Postgres query parser combines terms with **AND**. `websearch_to_tsquery` turns
`renewal notice period` into `'renew' & 'notic' & 'period'` — documents must contain all
three.

For a filter that is correct. For the keyword half of a hybrid search it is quietly
destructive: a four- or five-word question usually matches **nothing**, the keyword signal
contributes no ranking at all, and your hybrid search silently degrades into vector-only
search without reporting that anything went wrong.

In the demo corpus above, AND semantics return exactly **one** row. `pghybrid` OR-s the
terms by default, and precision comes back through ranking rather than exclusion:

```python
Config(..., text_match="any")   # default
Config(..., text_match="all")   # Postgres' native AND, if you want it
```

The naive way to do this — rewriting `&` into `|` inside a parsed tsquery — is wrong, and
`pghybrid` does not do it: `'a' & !'b'` becomes `'a' | !'b'`, which matches every document
that merely lacks `b`. Terms are tokenised and OR-combined individually, so quoted
`"phrases"` and `-negation` keep working.

## Find out which stage lost the answer

`explain` decomposes a single result set: both ranks, both raw scores, each signal's
contribution to the fused score, and — the part nobody else shows — the **near-miss band**,
the rows ranked just below your cut-off.

```
$ pghybrid explain "renewal notice period" --k 4 --near 3

  #  score     vector          text            document
  1  0.032258  rank 2  d.039   rank 2  0.300   Termination for convenience
  2  0.031319  rank 7  d.363   rank 1  0.600   Renewal pricing
  3  0.030835  rank 8  d.460   rank 2  0.300   Renewal terms
  4  0.030777  rank 6  d.275   rank 4  0.100   Notice requirements
  ─────────────────────────────── near miss ───────────────────────────────
  5  0.016393  rank 1  d.011      —       —    Automatic extension
  6  0.015873  rank 3  d.079      —       —    Subscription term
```

And when a document you *know* is in there does not come back:

```
$ pghybrid find "sixty days written notice"

  found in chunk 2 "Termination for convenience"
    vector rank   2 of 50 candidates
    text rank     2 of 50 candidates
    final rank    1
```

That single command separates *"the chunk was never indexed"* from *"the chunk was
outranked"* — two completely different bugs that look identical from the outside.

## Grade the index you already have

```
$ pghybrid doctor --dsn $DATABASE_URL --table chunks

  recall@10   0.68   ← measured against exact search, 50 sampled queries
```

`doctor` reads the real shape of your table and tells you what to change, with the
arithmetic shown so you can defend the choice:

- the index to create, and why — `lists = rows/1000` up to 1M rows and `sqrt(rows)` above
  it, a threshold that is very commonly applied on the wrong side
- **measured** recall@k against exact search, not a guess
- an `ef_search` / `probes` sweep so you pick your own point on the recall/latency curve
- queries silently falling back to a sequential scan, naming the filter that caused it
- filtered recall specifically — approximate indexes collapse under selective filters, which
  is where most multi-tenant applications live, and pgvector 0.8's
  [iterative index scans](https://github.com/pgvector/pgvector#iterative-index-scans) are
  the fix

It is read-only by default and will not write to your database without an explicit flag.

## The generated SQL

Nothing is hidden. `pghybrid` builds one statement, and you can always read it, copy it,
and stop using the package.

<details>
<summary><strong>Show the generated SQL</strong></summary>

See [`sql/hybrid_search.sql`](sql/hybrid_search.sql) for the same query, heavily commented
and ready to paste. In code:

```python
sql, params = search.build_query("renewal notice period", embedding=vec, limit=10)
print(sql)
```

Three things in it are load-bearing:

- **Filters go inside both candidate CTEs**, not after the fusion. Filtering afterwards
  throws away rows that were already ranked, so you ask for ten results and get four.
- **The join is `FULL OUTER`.** An inner join reduces hybrid search to the *intersection*
  of the two result sets, so a document only one signal found can never compete.
- **`ts_headline` runs only on the final page.** It re-parses the document text, so
  evaluating it inside a candidate CTE pays that cost for every candidate.

</details>

## Configuration

| option | default | what it does |
|---|---|---|
| `table`, `text_column`, `vector_column` | — | required |
| `id_column` | `"id"` | primary key |
| `tsvector_column` | `None` | a stored tsvector column. Omitted, the tsvector is computed inline: no migration needed, but no GIN index either |
| `language` | `"english"` | text search config. Use `"simple"` for identifiers, product codes and mixed-language corpora |
| `text_match` | `"any"` | `"any"` OR-s terms, `"all"` keeps Postgres' AND |
| `fusion` | `"rrf"` | or `"weighted"`, kept so `explain` can show you what it does |
| `k` | `60` | the RRF constant |
| `weights` | `1.0 / 1.0` | relative influence of each signal |
| `candidate_limit` | `50` | rows each signal contributes to the fusion |
| `metric` | `cosine` | `cosine`, `l2`, `inner_product`, `l1` |
| `vector_type` | `"vector"` | `"halfvec"` halves index size and build time, usually free on recall |
| `recency` | `None` | `Recency(column, half_life_days)` — exponential decay on the fused score |
| `paramstyle` | `"numeric"` | `$1` for asyncpg / node-postgres / raw SQL, `"pyformat"` (`%s`) for psycopg |
| `filter_columns` | `[]` | columns you may filter on; anything else is rejected rather than interpolated |

## Requirements

- PostgreSQL 12+
- `pgvector` 0.5+ (0.8+ recommended, for iterative index scans)
- Python 3.9+ / Node 18+

No other extension. That is the point, and it is asserted by a test that runs against a
stock `pgvector/pgvector:pg17` image with only `plpgsql` and `vector` installed.

## Prior art, credited

- **[ParadeDB `pg_search`](https://github.com/paradedb/paradedb)** — real BM25 in Postgres,
  and better at ranking than anything built on `ts_rank_cd`. Use it if you can install it.
- **[VectorChord](https://github.com/tensorchord/VectorChord)** and TigerData's
  **`pg_textsearch`** — also excellent, also extensions.
- **[`pg-hybrid-rrf`](https://github.com/BruceMong/pg-hybrid-rrf)** by Bruce Mong got to the
  scale-mismatch argument first and stated it clearly. The `effectiveWeights` idea in that
  README is a good one and this project's `explain` owes it a debt.
- The near-miss band is an idea worth stealing from every query planner ever written.

## Contributing

Bug reports and pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The test
suite runs against a real Postgres in Docker:

```bash
docker compose up -d
pytest                     # Python
npm test --prefix js       # TypeScript
```

## License

MIT © Pavan Gupta
