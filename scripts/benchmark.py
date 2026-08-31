"""Measure what the fusion costs, against a table you can rebuild.

The honest question a reader has is not "is this fast" but "what does adding the second
signal cost me". So this times the same query three ways — vector only, keyword only, and
both fused — over one corpus, and prints the difference.

Everything needed to reproduce the numbers is here. Nothing is quoted in the README that
this script did not print, and the README says which machine printed it, because a
latency measured on a laptop is not a latency measured on your database.

    docker compose up -d
    python scripts/benchmark.py                    # 100k rows, 384 dimensions
    python scripts/benchmark.py --rows 1000000 --dimensions 1536

The embeddings are random, which is the *worst* case for an approximate index: real
embeddings cluster, and clustered data is easier to search. Read these as an upper bound
on the fusion's overhead, not as a prediction of your own latency.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
import time
from typing import Any, Callable

import psycopg

from pghybrid import Config, HybridSearch
from pghybrid.sql import build_search_sql

DSN = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")
TABLE = "benchmark_chunks"

#: Size of the synthetic vocabulary. Corpus realism matters more here than it looks: an
#: early version of this script drew every document from a 36-word list, so every query
#: term matched essentially every row, ts_rank_cd had to score all 100,000 of them, and
#: keyword search measured 283ms. That number said nothing about Postgres and everything
#: about the fixture. Real vocabularies are Zipfian and a term matches a small fraction
#: of documents, which is what this reproduces.
VOCABULARY_SIZE = 5_000
WORDS_PER_DOCUMENT = (18, 40)

#: Ranks in the Zipf distribution to build queries from. Words this far down the tail
#: appear in a few percent of documents, which is the selectivity a real search term has.
QUERY_TERM_RANKS = [
    (140, 380, 900),
    (210, 640, 1500),
    (95, 470, 1100),
    (300, 820, 1900),
    (175, 550, 1300),
]


def build_vocabulary(size: int) -> tuple[list[str], list[float]]:
    """Pronounceable nonsense words plus Zipf weights, so frequencies look like language."""
    consonants, vowels = "bcdfgklmnprstvz", "aeiou"
    words: list[str] = []
    for index in range(size):
        syllables = 2 + (index % 2)
        word = "".join(
            consonants[(index * (syllable + 3)) % len(consonants)]
            + vowels[(index * (syllable + 1) + syllable) % len(vowels)]
            for syllable in range(syllables)
        )
        words.append(f"{word}{index}")
    weights = [1.0 / (rank + 1) for rank in range(size)]
    return words, weights


def build_corpus(
    connection: Any, rows: int, dimensions: int, seed: int = 7
) -> tuple[list[str], list[float]]:
    """Create and populate the benchmark table, then index it."""
    random.seed(seed)
    words, weights = build_vocabulary(VOCABULARY_SIZE)
    print(f"building {rows:,} rows x {dimensions} dimensions ...", flush=True)

    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
    connection.execute(
        f"""
        CREATE TABLE {TABLE} (
            id bigserial PRIMARY KEY,
            content text NOT NULL,
            embedding vector({dimensions}),
            fts tsvector GENERATED ALWAYS AS
                (to_tsvector('english', coalesce(content, ''))) STORED
        )
        """
    )

    started = time.perf_counter()
    with connection.cursor().copy(f"COPY {TABLE} (content, embedding) FROM STDIN") as copy:
        for _ in range(rows):
            document = random.choices(words, weights=weights, k=random.randint(*WORDS_PER_DOCUMENT))
            vector = "[" + ",".join(f"{random.uniform(-1, 1):.5f}" for _ in range(dimensions)) + "]"
            copy.write_row((" ".join(document), vector))
    print(f"  inserted in {time.perf_counter() - started:.1f}s", flush=True)

    for label, statement in (
        ("GIN on fts", f"CREATE INDEX ON {TABLE} USING gin (fts)"),
        (
            "HNSW on embedding",
            f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops)",
        ),
    ):
        started = time.perf_counter()
        connection.execute(statement)
        print(f"  {label} built in {time.perf_counter() - started:.1f}s", flush=True)

    connection.execute(f"ANALYZE {TABLE}")
    return words, weights


def time_modes(
    modes: dict[str, Callable[[int], Any]], runs: int, warmup: int
) -> dict[str, tuple[float, float]]:
    """Time every mode, interleaved, and return (p50, p95) milliseconds for each.

    Interleaving is the whole point. Run the modes in blocks instead and whichever goes
    last inherits a warm cache from the ones before it: measured that way, hybrid came
    out *faster* than keyword-only, which is impossible — hybrid runs the same keyword
    CTE plus a vector one. The ordering was the finding, not the query.

    Interleaving in a *fixed* order is not enough either, for the same reason one step
    down: whichever mode always runs second inherits the pages the first just touched.
    The order is shuffled per iteration so no mode has a systematic neighbour.
    """
    order = list(modes)
    samples: dict[str, list[float]] = {label: [] for label in modes}

    shuffler = random.Random(1234)

    for index in range(warmup):
        shuffler.shuffle(order)
        for label in order:
            modes[label](index)

    for index in range(runs):
        shuffler.shuffle(order)
        for label in order:
            started = time.perf_counter()
            modes[label](index)
            samples[label].append((time.perf_counter() - started) * 1000)

    results: dict[str, tuple[float, float]] = {}
    for label, values in samples.items():
        values.sort()
        p95_index = min(len(values) - 1, int(len(values) * 0.95))
        results[label] = (statistics.median(values), values[p95_index])
    return results


def server_times(
    connection: Any, statements: dict[str, tuple[str, list[Any]]], runs: int
) -> dict[str, float]:
    """Median server-side execution time per mode, from EXPLAIN ANALYZE.

    Worth reporting alongside the wall-clock figures because the two disagree, and the
    disagreement is informative. Keyword-only measures slower than hybrid end to end
    while the server says the opposite — which it should, since hybrid runs the same
    keyword CTE plus a vector one. The difference sits outside the server: not in
    planning, row count, payload size or result types, all of which were checked. Until
    someone explains it, the server column is the one that describes the query.
    """
    medians: dict[str, float] = {}
    for label, (sql, params) in statements.items():
        samples: list[float] = []
        for _ in range(runs):
            # This connection uses a mapping row factory, so the single EXPLAIN column
            # is reached by value rather than by position.
            plan = "\n".join(
                str(next(iter(row.values())))
                for row in connection.execute("EXPLAIN (ANALYZE) " + sql, params)
            )
            found = re.search(r"Execution Time: ([0-9.]+)", plan)
            if found:
                samples.append(float(found.group(1)))
        if samples:
            medians[label] = statistics.median(samples)
    return medians


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--keep", action="store_true", help="do not drop the table afterwards")
    parser.add_argument("--reuse", action="store_true", help="skip building, use what is there")
    args = parser.parse_args()

    random.seed(11)
    vectors = [
        [random.uniform(-1, 1) for _ in range(args.dimensions)]
        for _ in range(len(QUERY_TERM_RANKS))
    ]

    with psycopg.connect(DSN, autocommit=True, row_factory=psycopg.rows.dict_row) as connection:
        connection.execute("SET statement_timeout = 0")
        connection.execute("SET maintenance_work_mem = '512MB'")
        words, _ = build_vocabulary(VOCABULARY_SIZE)
        if not args.reuse:
            words, _ = build_corpus(connection, args.rows, args.dimensions)

        queries = [" ".join(words[rank] for rank in ranks) for ranks in QUERY_TERM_RANKS]

        server = connection.execute("SELECT version() AS v").fetchone()["v"].split(",")[0]
        pgvector = connection.execute(
            "SELECT extversion AS v FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()["v"]
        count = connection.execute(f"SELECT count(*) AS n FROM {TABLE}").fetchone()["n"]

        search = HybridSearch(
            Config(
                table=TABLE,
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                paramstyle="pyformat",
            ),
            execute=lambda sql, params: connection.execute(sql, params).fetchall(),
        )

        modes = {
            "vector only": lambda i: search.search(
                None, embedding=vectors[i % len(queries)], limit=args.limit
            ),
            "keyword only": lambda i: search.search(queries[i % len(queries)], limit=args.limit),
            "hybrid (both)": lambda i: search.search(
                queries[i % len(queries)],
                embedding=vectors[i % len(queries)],
                limit=args.limit,
            ),
        }

        print()
        print(f"  {server}, pgvector {pgvector}")
        print(f"  {count:,} rows x {args.dimensions} dimensions, top {args.limit}")
        print(
            f"  {args.runs} interleaved runs in shuffled order after {args.warmup} warmups, "
            f"{len(queries)} rotating queries"
        )
        print()
        timings = time_modes(modes, args.runs, args.warmup)

        statements = {
            "vector only": build_search_sql(
                search.config, embedding=vectors[0], text=None, limit=args.limit
            ),
            "keyword only": build_search_sql(
                search.config, embedding=None, text=queries[0], limit=args.limit
            ),
            "hybrid (both)": build_search_sql(
                search.config, embedding=vectors[0], text=queries[0], limit=args.limit
            ),
        }
        server = server_times(connection, statements, max(20, args.runs // 10))

        print(f"  {'mode':<16}{'p50':>10}{'p95':>10}{'server p50':>14}")
        print("  " + "-" * 50)
        for label, (p50, p95) in timings.items():
            srv = server.get(label)
            srv_text = f"{srv:>11.2f}ms" if srv is not None else f"{'-':>13}"
            print(f"  {label:<16}{p50:>9.2f}ms{p95:>9.2f}ms{srv_text}")

        vector_p50 = timings["vector only"][0]
        hybrid_p50 = timings["hybrid (both)"][0]
        overhead = hybrid_p50 - vector_p50
        print()
        print(
            f"  Adding the keyword signal costs {overhead:+.2f}ms at p50 "
            f"({overhead / vector_p50 * 100:+.0f}% over vector-only)."
        )
        print(
            "  The first two columns are wall-clock from the client, so they include a\n"
            "  round trip; the last is what the server reports for the same statement.\n"
            "  Embeddings here are random, which is the worst case for an approximate\n"
            "  index; real embeddings cluster and search faster."
        )

        if not args.keep and not args.reuse:
            connection.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
            print("\n  benchmark table dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
