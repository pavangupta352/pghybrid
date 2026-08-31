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
  The parity check compares 51 fixtures byte for byte, and the rejections too: an input
  one package builds a statement for and the other refuses is invisible to a check that
  only compares SQL.

- Driver adapters that also pin the placeholder style: `for_psycopg`, `for_sqlalchemy`,
  `for_asyncpg`, `for_django`; `forPg`, `forPostgresJs`, `forDrizzle`, `forKysely`.
- Guides for the audiences this replaces something for: Supabase, Neon after `pg_search`
  was removed for new projects, and pgai after it was archived.
- A LangChain retriever, as a tested example rather than a dependency.
- `py.typed`, so a consumer's type checker sees the annotations.

### Fixed
- An exclusion (`-term`) now constrains **both** signals. It was applied only to the
  tsquery, so the vector half still returned the excluded rows: they left the text
  candidates, arrived with a vector rank and no text rank, and RRF paid the best vector
  hit `1/(k+1)` — the largest single contribution available. A row you typed `-pricing`
  to be rid of could come back first. Found before release; the keyword-only test that
  covered it passed throughout, because the tsquery is exactly where the exclusion was
  already correct.
- A query of only exclusions no longer inverts the corpus. `!'pricing'` matches almost
  every row and `ts_rank_cd` scores a pure negation identically for all of them, so the
  keyword half was contributing an arbitrary order at full weight. It is dropped
  instead, the exclusion still applies, and with no embedding to fall back on the call
  says what is missing rather than returning a list ranked by nothing.

- Pagination past the candidate pool no longer returns an empty page. `candidate_limit`
  bounds the whole result set, so with the default 50 and a page size of 10, page 6 came
  back empty on a table where 490 rows matched — indistinguishable from having reached
  the end. It now raises, naming the pool size needed.

  The pool deliberately does **not** grow to cover the offset, which was the first fix
  and was worse than the bug: ranks are assigned inside the pool, so a pool that widens
  per page reorders every page. Paging 8×10 that way returned 71 distinct rows instead of
  80 and never showed 9 rows that a single `limit=80` query returns — duplicates and gaps
  in a search UI, with nothing to indicate anything was wrong.

- `doctor` checks that `id_column` is unique. The fusion joins on it twice, so a repeated
  id multiplies rows: pointing `id_column` at a `doc_id` over chunked text returned the
  same document ten times for `limit=10`, silently. A unique index is proof and produces
  no finding; without one, an actual duplicate is an error naming it, and no duplicate is
  a warning, because the first insert makes it wrong.
- `doctor` measures whether a hand-maintained `tsvector` still matches its text instead
  of warning that it might not. A stale column is silent in both directions — rows are
  returned for words they no longer contain and missing for the words they do — and reads
  as a relevance problem. A column that disagrees on every sampled row is reported
  separately as a text search configuration mismatch, which has a different fix. The
  sample is random rather than a bare `LIMIT`: an UPDATE writes the new row version at the
  end of the heap, so a `LIMIT` reads mostly the rows that were never touched and reported
  3 of 100 on a table that was 60 of 200.

### Security
- `language`, `query_parser` and `rank_function` are validated. They are interpolated
  rather than bound, and were unchecked, so a `Config` built from user input was an
  injection surface. Found and fixed before release.

### Notes on defaults
- `recency` reranks the candidate pool; it does not retrieve. Both signals choose their
  top `candidate_limit` rows on relevance alone and the decay is applied afterwards, so a
  row published today that no signal ranked highly cannot surface at any half-life.
  Measured: with a one-day half-life on 300 rows, a row published today at relevance rank
  250 is invisible at `candidate_limit=20` and first at `candidate_limit=300`. Retrieving
  on recency would mean a third candidate set ordered by timestamp, which returns recent
  rows nobody searched for.
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
