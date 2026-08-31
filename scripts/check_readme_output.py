"""Fail the build when the README shows output the tool no longer produces.

Sample output in a README rots quietly. Twice during this project's first day the
renderer changed and the README kept showing the old columns, which is exactly the kind
of small dishonesty that costs a reader's trust the first time they run the command and
see something different.

So the blocks are regenerated here and compared character for character. A failure means
one of two things, and the fix differs: either the renderer changed and the README needs
re-capturing, or the renderer changed in a way nobody intended.

    python scripts/check_readme_output.py            # verify
    python scripts/check_readme_output.py --update   # re-capture after a deliberate change
"""

from __future__ import annotations

import argparse
import difflib
import os
import pathlib
import re
import sys

import psycopg

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DSN = os.environ.get(
    "PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid"
)

sys.path.insert(0, str(ROOT / "scripts"))


def _connect():
    return psycopg.connect(DSN, row_factory=psycopg.rows.dict_row)


def render_explain() -> str:
    """The decomposition table, exactly as `pghybrid explain` prints it."""
    from seed_demo import DEMO_QUERY, query_vector

    from pghybrid import Config, HybridSearch, explain

    with _connect() as conn:
        search = HybridSearch(
            Config(
                table="chunks",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                extra_columns=["title"],
                paramstyle="pyformat",
            ),
            execute=lambda sql, params: conn.execute(sql, params).fetchall(),
        )
        text = explain(
            search,
            DEMO_QUERY,
            query_vector(),
            limit=4,
            near_miss=3,
            label_column="title",
        ).to_text()
    return text[: text.index("  effective weights")].rstrip()


def render_weights() -> str:
    """The effective-weights table.

    Deliberately uses a query whose terms reach every row, so both signals have full
    coverage and the measurement is about scale alone. The demo query does not, and
    conflating the two effects is the mistake this table exists to prevent.
    """
    from seed_demo import query_vector

    from pghybrid import Config, HybridSearch, Weights, explain

    query = "agreement party notice date term renewal period supplier customer"
    with _connect() as conn:
        search = HybridSearch(
            Config(
                table="chunks",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                extra_columns=["title"],
                paramstyle="pyformat",
                weights=Weights(vector=0.7, text=0.3),
            ),
            execute=lambda sql, params: conn.execute(sql, params).fetchall(),
        )
        text = explain(
            search, query, query_vector(), limit=3, near_miss=0, label_column="title"
        ).to_text()
    return text[text.index("  effective weights") :].rstrip()


#: (name, regex capturing the block body, renderer). The regex must have the text before
#: the block as group 1 and the text after it as group 2.
BLOCKS = [
    (
        "explain",
        re.compile(
            r'(\$ pghybrid explain "renewal notice period" --limit 4 --near-miss 3\n\n)'
            r".*?(\n```)",
            re.S,
        ),
        render_explain,
    ),
    (
        "effective weights",
        re.compile(r"(  effective weights).*?(\n```)", re.S),
        render_weights,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update", action="store_true", help="re-capture the blocks into the README"
    )
    args = parser.parse_args()

    from seed_demo import main as seed

    seed()

    readme = README.read_text()
    failures = 0

    for name, pattern, render in BLOCKS:
        match = pattern.search(readme)
        if match is None:
            print(
                f"FAIL: could not find the {name} block in README.md", file=sys.stderr
            )
            failures += 1
            continue

        actual = render()
        # The weights block starts with the text the regex captured as group 1, so the
        # comparison has to include it.
        current = match.group(0)[len(match.group(1)) : -len(match.group(2))].rstrip()
        expected = (
            actual[len(match.group(1)) :].rstrip()
            if name == "effective weights"
            else actual
        )

        if current == expected:
            print(f"ok    {name}")
            continue

        if args.update:
            readme = (
                readme[: match.start()]
                + match.group(1)
                + expected
                + match.group(2)
                + readme[match.end() :]
            )
            print(f"updated  {name}")
            continue

        failures += 1
        print(
            f"\nFAIL: the {name} block in README.md is not what the tool prints",
            file=sys.stderr,
        )
        diff = difflib.unified_diff(
            current.splitlines(),
            expected.splitlines(),
            "README.md",
            "actual",
            lineterm="",
        )
        for line in list(diff)[:24]:
            print(f"  {line}", file=sys.stderr)
        print(
            "\n  Re-capture with: python scripts/check_readme_output.py --update",
            file=sys.stderr,
        )

    if args.update:
        README.write_text(readme)
        return 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
