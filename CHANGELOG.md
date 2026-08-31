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
- Standalone SQL in `sql/`, usable without installing the package.
- Python and TypeScript packages, generating identical SQL from a shared golden snapshot.

### Notes on defaults
- Query terms are OR-ed (`text_match="any"`). Postgres' parsers AND by default, which
  makes multi-word queries match nothing and silently reduces hybrid search to
  vector-only search.
- Placeholder style is explicit (`paramstyle`), because `$1` and `%s` are not
  interchangeable and guessing wrong fails confusingly.
