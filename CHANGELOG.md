# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

First release.

### Added
- Hybrid search over `pgvector` + built-in full-text search, fused by Reciprocal Rank
  Fusion, generated as a single statement with one candidate CTE per signal.
- `explain()` — per-signal decomposition of a result set, a near-miss band showing the
  rows just below the cut-off, and an effective-weights measurement that shows how much
  influence each signal actually has as opposed to how much its weight claims.
- `find()` — locate text you expected to retrieve and report where it actually ranked,
  separating "never indexed" from "outranked".
- `doctor()` — index recommendations with the arithmetic shown, measured recall@k against
  exact search, an `ef_search`/`probes` sweep, sequential-scan detection, and filtered
  recall.
- `init` — schema introspection and migration generation.
- Recency decay, metadata filters scoped inside both candidate CTEs, `ts_headline`
  highlighting, `halfvec` support, and four distance metrics.
- Driver adapters that also set the placeholder style, so it cannot be got wrong:
  `for_psycopg`, `for_sqlalchemy`, `for_asyncpg`, `for_django` in Python; `forPg`,
  `forPostgresJs`, `forDrizzle`, `forKysely` in TypeScript. Each is tested against a
  live server and asserted to return identical rows, in identical order, with identical
  scores.
- Standalone SQL in `sql/`, usable without installing the package.
- Python and TypeScript packages, generating identical SQL from a shared golden snapshot.

- Driver adapters that also pin the placeholder style: `for_psycopg`, `for_sqlalchemy`,
  `for_asyncpg`, `for_django`; `forPg`, `forPostgresJs`, `forDrizzle`, `forKysely`.
- Guides for the audiences this replaces something for: Supabase, Neon after `pg_search`
  was removed for new projects, and pgai after it was archived.
- A LangChain retriever, as a tested example rather than a dependency.
- `py.typed`, so a consumer's type checker sees the annotations.

### Security
- `language`, `query_parser` and `rank_function` are validated. They are interpolated
  rather than bound, and were unchecked, so a `Config` built from user input was an
  injection surface. Found and fixed before release.

### Notes on defaults
- Query terms are OR-ed (`text_match="any"`). Postgres' parsers AND by default, which
  makes multi-word queries match nothing and silently reduces hybrid search to
  vector-only search.
- Placeholder style is explicit (`paramstyle`), because `$1` and `%s` are not
  interchangeable and guessing wrong fails confusingly.
- Query terms are capped (`max_query_terms`, default 200) and repeats collapsed. Past
  roughly 4,200 OR-ed terms Postgres reports a stack depth limit, which is not a useful
  thing to show someone who pasted a document into a search box.
- `weighted` fusion is available but its score is not a similarity: `1 - distance` assumes
  a bounded distance, which only cosine is. The ordering is correct for every metric.
