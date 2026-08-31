# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-08-31

### Fixed
- The CLI prints a sentence instead of a traceback for every refusal it already had a
  sentence for. The library raises `ValueError` with a message written for a person, and
  `main()` was not catching it, so "only excludes terms, so there is nothing to rank"
  arrived buried under fifteen stack frames. Server-side refusals (wrong vector
  dimensions, permissions, timeouts) are reported the same way, as `database error: ...`.
- Both builders refuse a non-finite embedding, naming the index. `json.loads` accepts
  `NaN` and `Infinity`, `float()` keeps them, and pgvector rejects them server-side with
  an error naming neither the argument nor the position. NaN was the dangerous one: had
  it ever got through, every comparison with it is false and the ordering would have been
  silently arbitrary rather than an error.
- Both builders refuse a negative `near_miss`, which flowed into the final `LIMIT` as
  `limit + near_miss` and produced a server error naming neither argument.
- `--label` selects the column it names. It used to be a printing detail, so labelling a
  column introspection had not already chosen printed `None` for every row, which reads
  as broken data. A label that names no column now says so and lists the columns; an
  operator-precedence slip that discarded an explicit `--label` whenever no extra columns
  were selected is also fixed.
- `init` on a table under a thousand rows shows arithmetic that is true: "500 rows / 1000
  rounds to 0, so 1 list (the minimum)" rather than "= 1 lists", which claimed a division
  that never produced it.

## [0.1.1] - 2026-08-31

### Security
- `highlight` escapes the document before `ts_headline` runs. The default delimiters are
  `<mark>`, so the result is meant to be rendered as HTML, which makes the surrounding
  document text active markup. Postgres does not escape it, and its parser removes only
  the tag shapes it recognises: `<script>alert(1)</script>` disappears, which makes the
  whole thing look safe, while `<img src=x onerror=alert(1)>` and `<svg/onload=alert(1)>`
  reach the caller intact, as does a bare `<` in ordinary text. Anyone rendering
  `result.highlight` had an injection path through their own stored documents.
  `Config.escape_highlight` (`escapeHighlight` in TypeScript) defaults to true, and can be
  turned off for delimiters that are not HTML.

  Only the three escaped characters change, so matching and mark placement are identical.
  Nothing else changed, and upgrading needs no code change.

## [0.1.0] - 2026-08-31

First release.

### Added
- Hybrid search over `pgvector` + built-in full-text search, fused by Reciprocal Rank
  Fusion, generated as a single statement with one candidate CTE per signal.
- `explain()`, per-signal decomposition of a result set, a near-miss band showing the
  rows just below the cut-off, and an effective-weights measurement that shows how much
  influence each signal actually has as opposed to how much its weight claims.
- `find()`, locate text you expected to retrieve and report where it actually ranked,
  separating "never indexed" from "outranked".
- `doctor()`, index recommendations with the arithmetic shown, measured recall@k against
  exact search, an `ef_search`/`probes` sweep, sequential-scan detection, and filtered
  recall.
- `init`, schema introspection and migration generation.
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
  hit `1/(k+1)`, the largest single contribution available. A row you typed `-pricing`
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
  back empty on a table where 490 rows matched, indistinguishable from having reached
  the end. It now raises, naming the pool size needed.

  The pool deliberately does **not** grow to cover the offset, which was the first fix
  and was worse than the bug: ranks are assigned inside the pool, so a pool that widens
  per page reorders every page. Paging 8×10 that way returned 71 distinct rows instead of
  80 and never showed 9 rows that a single `limit=80` query returns, duplicates and gaps
  in a search UI, with nothing to indicate anything was wrong.

- Both README quickstarts were wrong, and the documentation's code is now run in CI rather than only read, the README and both migration guides. The
  Python one printed `None` for every title, because it asked for `row.get("title")` from
  a config that never selected `title`, a new reader's first run looked like the library
  was broken. The TypeScript one imported `Config`, a type-only export, as a value, which
  compiles until the reader's tsconfig sets `verbatimModuleSyntax`, the setting TypeScript
  5 recommends, and then fails on line 1. `scripts/check_docs_code.py` executes the
  Python blocks against tables the guides describe and typechecks the TypeScript ones
  against the packed tarball.
- `init --apply` no longer reports success for work it did not do. A bare `vector` column
  has to be given the dimension the model produces, which nothing here can know, so the
  migration carries that as a comment with a placeholder, and `--apply` sent it to the
  server, which accepts a comment as an empty command, printed `ok` for it and ended with
  "done." at exit code 0. The column was untouched. Statements that are entirely comments
  are now listed as work still to do by hand, and the command exits 1 while any remain.
- The Supabase guide's `hybrid_search` function is executed in CI. It is a hand-written
  copy of the query this library generates, which makes it the documentation most likely
  to drift, the generated SQL is checked from every direction and nothing at all ran
  that function. Five assertions cover it, including that a row found by both signals
  scores above `1/(k+1)`, which is what catches a fusion reduced to one signal or an RRF
  numerator written as an integer.
- `explain(find=...)` names an exclusion as the reason a row is missing. It used to see a
  row that was #1 on both signals and absent from the result, conclude it had lost the
  fused ordering, and advise raising `candidate_limit`, a knob that can never bring back
  a row the query itself removed. The real reason was the `-term` in the query string the
  caller had just typed.
- A column selected through to the result can no longer be named like one the statement
  computes. Postgres permits two output columns with the same name and the driver keeps
  the last, which is the table's, so the computed value disappeared without an error:
  listing a column called `text_rank` turned `text_rank=1, matched_by="both"` into
  `text_rank=None, matched_by="vector"` on the same query and the same data, the library
  reporting that the keyword signal missed rows it had ranked first. `Config` now refuses
  the whole reserved set, and a test reads the aliases back out of a fully-featured
  statement so the list cannot fall behind the query.
- The `tsvector` check covers generated columns. Scoping it to hand-maintained ones looked
  obviously right, a generated column cannot fall behind its own expression, and missed
  the case where the expression is not the one the config describes. A column generated
  with `'english'` and searched by a config saying `'simple'` returned 5 rows against 0 on
  the same table, and nothing reported it. The finding quotes the stored expression, and a
  generated column is never described as stale, which it cannot be.
- `doctor` checks that `id_column` is unique. The fusion joins on it twice, so a repeated
  id multiplies rows: pointing `id_column` at a `doc_id` over chunked text returned the
  same document ten times for `limit=10`, silently. A unique index is proof and produces
  no finding; without one, an actual duplicate is an error naming it, and no duplicate is
  a warning, because the first insert makes it wrong.
- `doctor` measures whether a hand-maintained `tsvector` still matches its text instead
  of warning that it might not. A stale column is silent in both directions, rows are
  returned for words they no longer contain and missing for the words they do, and reads
  as a relevance problem. A column that disagrees on every sampled row is reported
  separately as a text search configuration mismatch, which has a different fix. The
  sample is random rather than a bare `LIMIT`: an UPDATE writes the new row version at the
  end of the heap, so a `LIMIT` reads mostly the rows that were never touched and reported
  3 of 100 on a table that was 60 of 200.

### Security
- `highlight` escapes the document before `ts_headline` runs. The default delimiters are
  `<mark>`, so the result is meant to be rendered as HTML, which makes the surrounding
  document text active markup. Postgres does not escape it, and its parser drops only the
  tag shapes it recognises: `<script>alert(1)</script>` disappears, which makes the whole
  thing look safe, while `<img src=x onerror=alert(1)>` and `<svg/onload=alert(1)>` reach
  the caller intact. Anyone rendering `result.highlight` had an injection path through
  their own stored documents. `Config.escape_highlight` defaults to true and can be turned
  off for non-HTML delimiters.
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
