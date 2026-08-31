"""Run the files in sql/ against a real database, exactly as a user would.

The standalone SQL is a supported way to use this project — the README tells people
they can paste it into the Supabase editor and never install the package — so it is
tested like code rather than trusted like documentation. Documentation that is only
read goes stale silently; this fails the build instead.

The only edit made to the files is the table rename that a user would also make, so
a passing run means the committed text works unmodified apart from the identifiers
that are explicitly marked CHANGE ME.
"""

from __future__ import annotations

import os
import pathlib
import sys

import psycopg

ROOT = pathlib.Path(__file__).resolve().parent.parent
DSN = os.environ.get(
    "PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid"
)

# The demo corpus is built so the planted answer ranks second on both signals and
# first on neither. If the standalone SQL returns it first, the fusion in these files
# is doing the same thing the packages do.
EXPECTED_TOP_TITLE = "Termination for convenience"
DEMO_QUERY = "renewal notice period"
QUERY_VECTOR = "[1,0,0,0,0,0,0,0]"

TABLE = "standalone_check"


def load(name: str) -> str:
    """Read a sql/ file, applying only the rename a real user would apply."""
    return (ROOT / "sql" / name).read_text().replace("chunks", TABLE)


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from seed_demo import DOCUMENTS, to_pgvector, unit_vector  # noqa: E402

    # EXECUTE is a utility statement, so Postgres cannot infer parameter types for it
    # over the extended protocol. Binding client-side sends literals instead, which is
    # also closer to what someone pasting this into a SQL editor actually does.
    with psycopg.connect(
        DSN, autocommit=True, cursor_factory=psycopg.ClientCursor
    ) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # A table with no tsvector column and no indexes, so migration.sql has to do
        # real work rather than skipping everything it finds already present.
        conn.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
        conn.execute(
            f"""
            CREATE TABLE {TABLE} (
                id         bigserial PRIMARY KEY,
                tenant_id  integer NOT NULL DEFAULT 1,
                title      text NOT NULL,
                content    text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                embedding  vector(8)
            )
            """
        )
        for angle, title, content in DOCUMENTS:
            conn.execute(
                f"INSERT INTO {TABLE} (title, content, embedding) VALUES (%s, %s, %s)",
                (title, content, to_pgvector(unit_vector(angle))),
            )

        conn.execute(load("migration.sql"))
        print("migration.sql applied to a table that had no tsvector column")

        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = %s", (TABLE,)
            )
        }
        for expected in (f"{TABLE}_fts_idx", f"{TABLE}_embedding_idx"):
            if expected not in indexes:
                print(f"FAIL: migration.sql did not create {expected}", file=sys.stderr)
                return 1
        print(f"indexes present: {', '.join(sorted(indexes))}")

        search_sql = load("hybrid_search.sql").rstrip().rstrip(";")
        conn.execute(
            f"PREPARE standalone (vector, text, int, int) AS {search_sql}"
        )
        cursor = conn.execute(
            "EXECUTE standalone(%s, %s, %s, %s)", (QUERY_VECTOR, DEMO_QUERY, 50, 5)
        )
        columns = [d.name for d in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            print("FAIL: hybrid_search.sql returned no rows", file=sys.stderr)
            return 1

        title_index = columns.index("title")
        top_title = rows[0][title_index]
        print(f"hybrid_search.sql returned {len(rows)} rows; top result {top_title!r}")

        if top_title != EXPECTED_TOP_TITLE:
            print(
                f"FAIL: expected {EXPECTED_TOP_TITLE!r} to rank first, got {top_title!r}."
                " The fusion in sql/hybrid_search.sql has drifted from the packages.",
                file=sys.stderr,
            )
            return 1

        # A query of nothing but stop words yields a NULL tsquery. The text signal must
        # contribute nothing while the vector side still answers, rather than the whole
        # statement failing.
        stopword_rows = conn.execute(
            "EXECUTE standalone(%s, %s, %s, %s)",
            (QUERY_VECTOR, "the and of", 50, 3),
        ).fetchall()
        if len(stopword_rows) != 3:
            print(
                f"FAIL: a stop-word-only query returned {len(stopword_rows)} rows,"
                " expected the vector signal to answer alone",
                file=sys.stderr,
            )
            return 1
        print("stop-word-only query degrades to vector-only, as intended")

        conn.execute("DEALLOCATE standalone")
        conn.execute(f"DROP TABLE {TABLE} CASCADE")

    print("\nsql/ files run unmodified against a stock pgvector image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
