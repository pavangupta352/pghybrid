"""Reproduce the comparison at the top of the README.

Runs the same query three ways against the same corpus — vector only, keyword only,
and both fused — and prints the three rankings side by side.

    docker compose up -d
    pip install "pghybrid[cli]"
    python examples/three_way_comparison.py

The corpus is twelve clauses from a software contract. The query is a question
somebody would actually type, and the clause that answers it never uses the words in
the question. Neither signal ranks it first. Fusion does.

Embeddings here are placed by hand on a circle rather than produced by a model, so the
example needs no API key, no network and no GPU, and gives the same answer on every
machine. The geometry stands in for semantic similarity; the keyword side is real
Postgres full-text search over real English.
"""

from __future__ import annotations

import os
import sys

import psycopg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from seed_demo import DEMO_QUERY, PLANTED_TITLE, query_vector  # noqa: E402
from seed_demo import main as seed

from pghybrid import Config  # noqa: E402
from pghybrid.search import HybridSearch  # noqa: E402

DSN = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")
ROWS = 5


def main() -> int:
    seed()

    connection = psycopg.connect(DSN, row_factory=psycopg.rows.dict_row)
    search = HybridSearch(
        Config(
            table="chunks",
            text_column="content",
            vector_column="embedding",
            tsvector_column="fts",
            extra_columns=["title"],
            paramstyle="pyformat",
        ),
        execute=lambda sql, params: connection.execute(sql, params).fetchall(),
    )

    embedding = query_vector()
    columns = {
        "vector only": search.search(None, embedding=embedding, limit=ROWS),
        "keyword only": search.search(DEMO_QUERY, limit=ROWS),
        "pghybrid": search.search(DEMO_QUERY, embedding=embedding, limit=ROWS),
    }

    print(f"\n  query: {DEMO_QUERY!r}")
    print(f"  the clause that answers it: {PLANTED_TITLE!r}")
    print("  (it says 'sixty days written notice' and never says 'renewal')\n")

    width = 30
    header = "  # " + "".join(name.ljust(width + 4) for name in columns)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for position in range(ROWS):
        cells = []
        for results in columns.values():
            if position < len(results):
                title = str(results[position].get("title"))
                marker = " *" if title == PLANTED_TITLE else "  "
                cells.append((title[:width]).ljust(width) + marker + "  ")
            else:
                cells.append("—".ljust(width) + "    ")
        print(f"  {position + 1} " + "".join(cells))

    print("\n  * marks the clause that actually answers the question\n")

    for name, results in columns.items():
        top = str(results[0].get("title")) if results else "nothing"
        verdict = "found it" if top == PLANTED_TITLE else f"returned {top!r}"
        print(f"    {name:<14} {verdict}")

    hybrid_top = str(columns["pghybrid"][0].get("title"))
    if hybrid_top != PLANTED_TITLE:
        print("\n  fusion did not surface the planted answer", file=sys.stderr)
        return 1

    print(
        "\n  Neither signal alone ranked it first. It was second on both, and "
        "\n  Reciprocal Rank Fusion is what moved it to the top.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
