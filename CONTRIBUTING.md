# Contributing

Thanks for considering it. Bug reports are as welcome as pull requests.

## Getting set up

```bash
git clone https://github.com/pavangupta352/pghybrid
cd pghybrid
docker compose up -d                 # Postgres 17 + pgvector on port 55432

# Python
cd python && uv venv && uv pip install -e ".[dev]" && python -m pytest

# TypeScript
cd js && npm install && npm test
```

Tests marked `integration` need the database; the rest are pure functions of a `Config`
and run anywhere.

Before opening a pull request, run everything CI runs, in the order CI runs it:

```bash
scripts/check_all.sh          # or: scripts/check_all.sh fast, to skip the packaging build
```

A README code block is checked if the line before its fence is `<!-- check:python -->` or
`<!-- check:ts -->`, which renders as nothing on GitHub. Add the marker to any block a
reader is meant to copy and run; leave it off fragments that are illustrative rather than
runnable, like the list of four adapters assigning the same name four times.

Worth using rather than running the pieces you remember. `npm test` does not typecheck —
vitest strips the types — so a type error in a test file passes locally and fails CI, and
the parity, README-output and packaging checks are easy to forget entirely.

## What makes a change easy to accept

**The two implementations must agree.** Python and TypeScript generate the same SQL, and
`python/tests/golden/` holds the snapshots that prove it. A change to query generation
means updating both and regenerating the snapshot in the same pull request.

**Explain the failure mode, not the syntax.** The comments in this codebase say why a
decision exists — which bug it prevents, which surprise it avoids. A comment that
restates the line above it will be asked about in review.

**No runtime dependencies.** Neither package has any and neither should acquire one.
Drivers are injected by the caller; test-only dependencies belong in the dev extras.

**Correctness claims need a test.** Anything the README asserts about behaviour should
fail loudly if it stops being true. That is especially so for the claim that no extension
beyond `pgvector` is required, which is asserted by a test rather than by prose.

## Reporting a bug

The most useful report includes the `Config`, the generated SQL (`build_query` returns it),
your pgvector version, and what you expected to rank where. `pghybrid explain` output is
worth a thousand words of description.

## Scope

This is a query layer, deliberately. It does not embed text, chunk documents, rerank
results, or talk to any vector store other than Postgres. Proposals that would add those
are likely to be declined — not because they are bad ideas, but because staying small is
what keeps this auditable and installable.
