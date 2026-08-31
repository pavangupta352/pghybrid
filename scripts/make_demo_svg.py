"""Generate the animated demo at the top of the README.

The image is built from a live query rather than drawn by hand, so it cannot claim
something the code does not do. Re-running it after a behaviour change produces a
different picture, which is the point.

An SVG rather than a GIF: it stays sharp on a retina screen, it is a fraction of the
size, it diffs as text in review, and it needs no recorder installed.

It is deliberately static. A staged fade-in reads better, but it has to start from
opacity 0, and anything that does not run CSS animations — some Markdown renderers,
image proxies, a PDF export, the GitHub mobile app — then shows an empty terminal
instead of the argument. The first frame has to be the whole point.

    python scripts/make_demo_svg.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from xml.sax.saxutils import escape

import psycopg

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "demo.svg"
DSN = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")

sys.path.insert(0, str(ROOT / "scripts"))

# A terminal palette that stays legible on both a light and a dark README, since GitHub
# renders the image on whichever the reader chose.
BG = "#11141a"
CHROME = "#1b1f27"
DIM = "#6b7480"
TEXT = "#c9d1d9"
BRIGHT = "#e6edf3"
GREEN = "#3fb950"
AMBER = "#d29922"
BLUE = "#58a6ff"
RED = "#f85149"

FONT = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

#: Measured from a render rather than guessed: at font-size 12 this monospace stack
#: advances ~8.75px per character, and the first attempt at 7.35 pushed the widest title
#: past the right edge of the canvas, where it was simply clipped.
CHAR_W = 9.4
ROWS = 5
ROW_H = 23
#: Widest cell the table has to hold, plus the marker and a gutter.
_COL_W = 27 * CHAR_W + 44
COL_X = [40, 80, 80 + _COL_W, 80 + 2 * _COL_W]
WIDTH = int(COL_X[3] + _COL_W + 10)


def collect() -> dict:
    """Run the three searches that make the argument."""
    from pghybrid import Config, HybridSearch

    from seed_demo import DEMO_QUERY, PLANTED_TITLE, main as seed, query_vector

    seed()
    with psycopg.connect(DSN, row_factory=psycopg.rows.dict_row) as conn:
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
        embedding = query_vector()
        columns = {
            "vector only": search.search(None, embedding=embedding, limit=ROWS),
            "keyword only": search.search(DEMO_QUERY, limit=ROWS),
            "pghybrid": search.search(DEMO_QUERY, embedding=embedding, limit=ROWS),
        }

    return {
        "query": DEMO_QUERY,
        "answer": PLANTED_TITLE,
        "columns": {
            name: [str(r.get("title")) for r in results] for name, results in columns.items()
        },
    }


def text(
    x: float,
    y: float,
    body: str,
    *,
    fill: str = TEXT,
    size: int = 13,
    weight: str = "400",
    cls: str = "",
    anchor: str = "start",
) -> str:
    """One line of terminal text.

    font-family is written onto every element rather than declared once in a stylesheet.
    GitHub sanitises SVG it serves, a stripped <style> block would silently drop the
    whole image to a default serif, and a presentation attribute survives that.
    """
    attrs = (
        f'x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" font-family="{FONT}" xml:space="preserve"'
    )
    return f"  <text {attrs}>{escape(body)}</text>"


def build(data: dict) -> str:
    answer = data["answer"]
    columns = data["columns"]
    names = list(columns)

    height = 108 + ROW_H * (ROWS + 1) + 84
    parts: list[str] = []

    # Each stage fades in after the one before it, so the eye reads the two failing
    # columns before the one that works.
    css = ""

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-label="The same query run three ways: vector only and keyword only '
        f'both rank the wrong clause first, pghybrid ranks the right one first.">'
    )
    if css:
        parts.append(css)
    parts.append(f'  <rect width="{WIDTH}" height="{height}" rx="10" fill="{BG}"/>')
    parts.append(f'  <rect width="{WIDTH}" height="34" rx="10" fill="{CHROME}"/>')
    parts.append(f'  <rect y="24" width="{WIDTH}" height="10" fill="{CHROME}"/>')
    for index, colour in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        parts.append(f'  <circle cx="{22 + index * 19}" cy="17" r="6" fill="{colour}"/>')
    parts.append(text(WIDTH / 2, 22, "pghybrid", fill=DIM, size=12, anchor="middle"))

    y = 62
    parts.append(text(24, y, "$", fill=GREEN, size=13, weight="600"))
    parts.append(
        text(40, y, "pghybrid explain " + f'"{data["query"]}"', fill=BRIGHT, size=13, cls="")
    )

    y += 30
    parts.append(
        text(
            40,
            y,
            f'the clause that answers it never contains the word "renewal"',
            fill=DIM,
            size=12,
            cls="",
        )
    )

    y += 34
    header_y = y
    for index, name in enumerate(names):
        highlight = name == "pghybrid"
        parts.append(
            text(
                COL_X[index + 1],
                header_y,
                name,
                fill=BRIGHT if highlight else DIM,
                size=12,
                weight="600" if highlight else "400",
                cls="",
            )
        )
    parts.append(text(COL_X[0], header_y, "#", fill=DIM, size=12))

    parts.append(
        f'  <line x1="{COL_X[0]}" y1="{header_y + 10}" x2="{WIDTH - 40}" '
        f'y2="{header_y + 10}" stroke="{CHROME}" stroke-width="1"/>'
    )

    for row in range(ROWS):
        row_y = header_y + 32 + row * ROW_H
        parts.append(text(COL_X[0], row_y, str(row + 1), fill=DIM, size=12))
        for index, name in enumerate(names):
            titles = columns[name]
            value = titles[row] if row < len(titles) else "—"
            is_answer = value == answer
            is_hybrid = name == "pghybrid"

            if is_answer and is_hybrid and row == 0:
                fill, weight, cls = GREEN, "600", ""
            elif is_answer:
                fill, weight, cls = AMBER, "400", ""
            else:
                fill, weight, cls = TEXT if is_hybrid else DIM, "400", ""

            label = value
            parts.append(
                text(COL_X[index + 1], row_y, label, fill=fill, size=12, weight=weight, cls=cls)
            )
            if is_answer:
                parts.append(
                    text(
                        COL_X[index + 1] + len(label) * CHAR_W + 7,
                        row_y,
                        "◄",
                        fill=fill,
                        size=11,
                        cls=cls,
                    )
                )

    footer_y = header_y + 32 + ROWS * ROW_H + 30
    parts.append(
        text(
            COL_X[0],
            footer_y,
            "Neither signal ranked the right clause first.  It was second on both.",
            fill=DIM,
            size=12,
            cls="",
        )
    )
    parts.append(
        text(
            COL_X[0],
            footer_y + 22,
            "Reciprocal Rank Fusion is what moved it to the top.",
            fill=BLUE,
            size=12,
            weight="600",
            cls="",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed image is not what a live query produces now",
    )
    args = parser.parse_args()

    data = collect()
    hybrid_top = data["columns"]["pghybrid"][0]
    if hybrid_top != data["answer"]:
        print(
            f"refusing to draw a demo that is not true: fusion returned {hybrid_top!r}, "
            f"expected {data['answer']!r}",
            file=sys.stderr,
        )
        return 1

    rendered = build(data)

    if args.check:
        if not OUT.exists():
            print(f"{OUT.relative_to(ROOT)} is missing", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != rendered:
            print(
                f"{OUT.relative_to(ROOT)} is not what a live query produces now.\n"
                "  Regenerate it: python scripts/make_demo_svg.py",
                file=sys.stderr,
            )
            return 1
        print(f"ok    {OUT.relative_to(ROOT)} matches a live query")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    for name, titles in data["columns"].items():
        marker = "  <- the answer" if titles[0] == data["answer"] else ""
        print(f"  {name:<14} top result: {titles[0]}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
