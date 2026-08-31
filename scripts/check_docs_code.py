"""Run the code the documentation tells people to copy.

There is already a check that the README's *output* blocks match what the tool prints.
The input blocks had nothing, on the README or on the migration guides, and they rot the
same way — worse, because a reader runs them before they have any reason to trust the
project, and the guides address people who are mid-migration and least willing to forgive
a broken example.

Both quickstarts were broken when this was written, and each in a way no existing check
could see:

  * the Python block printed ``row.get("title")`` for a config that never selected
    ``title``, so every line of a new user's first run came back ``None``;
  * the TypeScript block wrote ``import { Config, HybridSearch }`` where ``Config`` is a
    type-only export, which is fine until the reader's tsconfig sets
    ``verbatimModuleSyntax`` — the setting TypeScript 5 recommends — and then it is a
    compile error on line 1.

Blocks are opted in with an HTML comment on the line before the fence, which renders as
nothing on GitHub:

    <!-- check:python -->    run it against the demo database
    <!-- check:ts -->        typecheck it against the built package

Fragments that are deliberately not runnable — four assignments to the same ``const`` to
show four adapters — simply carry no marker.

    docker compose up -d && python scripts/check_docs_code.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: Every page whose code a reader is expected to copy. The guides address people who are
#: migrating, which is the audience least willing to forgive a broken example.
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs" / "guides").glob("*.md"))]
FIXTURES = ROOT / "scripts" / "doc_fixtures.sql"
# CI runs Postgres on 5432 and docker-compose maps it to 55432 locally, so this has to
# come from the environment like every other script here does.
DSN = os.environ.get(
    "PGHYBRID_TEST_DSN",
    os.environ.get("PGHYBRID_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid"),
)

BLOCK = re.compile(
    r"<!-- check:(?P<kind>python|ts) -->\n```(?:python|ts)\n(?P<body>.*?)```", re.DOTALL
)

#: Names the Python blocks use without defining, because defining them would be noise on
#: a page someone is reading to learn. They are what a reader is expected to already have:
#: a connection, and a query vector from whatever model they use.
PYTHON_PREAMBLE = f'''
import psycopg
from psycopg.rows import dict_row

conn = psycopg.connect({DSN!r}, row_factory=dict_row, autocommit=True)
query_vector = vec = vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def embed(text):
    """Stands in for the reader's own model.

    The guides write `embed(query)` with a comment saying "whatever you already use",
    which is the honest thing to show and is not runnable. Supplying it here keeps the
    rest of those blocks — the part that is this project's API — under test.
    """
    return query_vector
'''

#: The same idea for TypeScript: ambient declarations so the block typechecks without a
#: live database or a driver installed.
TS_PREAMBLE = """
declare const pool: { query(sql: string, params: unknown[]): Promise<{ rows: unknown[] }> };
declare const embedding: number[];
"""

failures: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def blocks(kind: str) -> list[tuple[str, str]]:
    """Every marked block of one kind, with the page it came from."""
    found = []
    for page in PAGES:
        for match in BLOCK.finditer(page.read_text()):
            if match.group("kind") == kind:
                found.append((page.relative_to(ROOT).as_posix(), match.group("body")))
    return found


def load_fixtures() -> None:
    """Create the tables the guides' examples name, so they can be run rather than read."""
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute(FIXTURES.read_text())


def check_python(workspace: pathlib.Path) -> None:
    found = blocks("python")
    print(f"\nPython blocks ({len(found)})")
    if not found:
        report("at least one python block is checked", False, "no <!-- check:python --> markers")
        return
    for index, (page, body) in enumerate(found, 1):
        script = workspace / f"readme_{index}.py"
        script.write_text(PYTHON_PREAMBLE + body)
        result = subprocess.run(
            [str(ROOT / "python" / ".venv" / "bin" / "python"), str(script)],
            capture_output=True,
            text=True,
        )
        first = f"{page}: {body.strip().splitlines()[0][:44]}"
        if result.returncode != 0:
            report(first, False, result.stderr.strip().splitlines()[-1][:120])
            continue
        # Running without raising is not the bar. The Python quickstart ran perfectly and
        # printed `0.032258 None` for every row, because it asked for a column the config
        # never selected -- a first run that looks like the library is broken. So a
        # checked block may not print None at all: a quickstart exists to show something
        # working, and None is what a quietly-missing value looks like. A block that
        # genuinely needs to print None does not get a marker.
        printed = result.stdout.strip()
        if "None" in printed:
            report(first, False, f"printed None: {printed.splitlines()[0][:70]}")
            continue
        report(first, True, printed.splitlines()[0][:60] if printed else "no output")


def check_ts(workspace: pathlib.Path) -> None:
    found = blocks("ts")
    print(f"\nTypeScript blocks ({len(found)})")
    if not found:
        report("at least one ts block is checked", False, "no <!-- check:ts --> markers")
        return

    project = workspace / "consumer"
    project.mkdir()
    # Pack what the build produces, not whatever dist/ happens to hold. A CI job that
    # installs but does not build packs a package with no entry point, and every block
    # then fails with "cannot find module" for a reason that has nothing to do with the
    # README.
    subprocess.run(
        ["npm", "run", "--silent", "build"],
        cwd=ROOT / "js",
        capture_output=True,
        text=True,
        check=True,
    )
    packed = subprocess.run(
        ["npm", "pack", "--pack-destination", str(workspace)],
        cwd=ROOT / "js",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()[-1]
    (project / "package.json").write_text(json.dumps({"name": "c", "private": True, "type": "module"}))
    subprocess.run(
        ["npm", "install", "--silent", str(workspace / packed), "typescript"],
        cwd=project,
        capture_output=True,
        text=True,
        check=True,
    )
    # verbatimModuleSyntax is what TypeScript 5 recommends, and it is the setting that
    # turns a value import of a type into a compile error. Checking without it would have
    # passed the very block that was broken.
    (project / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "strict": True,
                    "target": "ES2022",
                    "module": "node16",
                    "moduleResolution": "node16",
                    "noEmit": True,
                    "verbatimModuleSyntax": True,
                    "skipLibCheck": True,
                }
            }
        )
    )

    for index, (page, body) in enumerate(found, 1):
        for stale in project.glob("*.ts"):
            stale.unlink()
        (project / f"readme_{index}.ts").write_text(TS_PREAMBLE + body)
        result = subprocess.run(["npx", "tsc"], cwd=project, capture_output=True, text=True)
        first = f"{page}: {body.strip().splitlines()[0][:44]}"
        report(first, result.returncode == 0, result.stdout.strip().splitlines()[0][:120]
               if result.returncode else "")


def main() -> int:
    print("Running the code the documentation tells people to copy.")
    load_fixtures()
    with tempfile.TemporaryDirectory() as directory:
        workspace = pathlib.Path(directory)
        check_python(workspace)
        if shutil.which("npm"):
            check_ts(workspace)
        else:
            print("\nnpm not on PATH, skipping the TypeScript blocks")

    print()
    if failures:
        print(
            f"{len(failures)} documentation block(s) do not work: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    print("Every checked documentation block runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
