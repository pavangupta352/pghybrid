"""Check what a user receives, not what the repository contains.

Everything else in CI runs against the working tree. That misses a whole class of
problem: a file that exists in the repo but never makes it into the artifact, an exports
map that resolves for the test runner and not for a consumer, a type declaration the
package advertises and does not ship.

One of those was real here. The Python package was annotated throughout, checked itself
under ``mypy --strict``, and declared the ``Typing :: Typed`` classifier — and shipped no
``py.typed`` marker, so every consumer's type checker reported every symbol as ``Any``.
Nothing in the repository could have caught it, because the repository was fine.

So this builds both artifacts, installs them into empty environments, and uses them the
way a stranger would.

    python scripts/check_packaging.py
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
failures: list[str] = []


def run(
    command: list[str], cwd: pathlib.Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        tail = f"{result.stdout[-1500:]}{result.stderr[-1500:]}"
        raise RuntimeError(f"{' '.join(command[:3])} failed in {cwd}:\n{tail}")
    return result


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def check_python(workspace: pathlib.Path) -> None:
    print("\nPython wheel")
    dist = workspace / "dist"
    run(["uv", "build", "--wheel", "--out-dir", str(dist)], ROOT / "python")
    wheels = list(dist.glob("*.whl"))
    report("wheel builds", bool(wheels), wheels[0].name if wheels else "nothing produced")
    if not wheels:
        return

    # PEP 561: without this file in the *artifact*, annotations are invisible downstream.
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    report("wheel contains py.typed", "pghybrid/py.typed" in names)

    venv = workspace / "consumer-venv"
    run(["uv", "venv", "-q", str(venv)], ROOT)
    python = venv / "bin" / "python"
    run(["uv", "pip", "install", "--python", str(python), "-q", str(wheels[0]), "mypy"], ROOT)

    consumer = workspace / "consumer.py"
    consumer.write_text(
        "from pghybrid import Config, HybridSearch\n"
        "cfg = Config(table='t', text_column='c', vector_column='e')\n"
        "reveal_type(cfg.k)\n"
        "reveal_type(HybridSearch)\n"
    )
    result = run([str(venv / "bin" / "mypy"), str(consumer)], workspace, check=False)
    saw_any = 'Revealed type is "Any"' in result.stdout
    saw_stub_warning = "py.typed marker" in result.stdout
    report(
        "a consumer's type checker sees real types",
        not saw_any and not saw_stub_warning,
        'reported "Any"' if saw_any else "",
    )

    entry = run(
        [str(python), "-c", "import pghybrid; print(pghybrid.__version__)"], workspace, check=False
    )
    report("imports from a clean environment", entry.returncode == 0, entry.stdout.strip())

    cli = run([str(venv / "bin" / "pghybrid"), "--version"], workspace, check=False)
    report("console script works", cli.returncode == 0, cli.stdout.strip())


def check_npm(workspace: pathlib.Path) -> None:
    print("\nnpm tarball")
    js = ROOT / "js"
    run(["npm", "run", "build"], js)
    packed = (
        run(["npm", "pack", "--pack-destination", str(workspace)], js).stdout.strip().splitlines()
    )
    tarball = workspace / packed[-1]
    report("tarball builds", tarball.exists(), tarball.name)
    if not tarball.exists():
        return

    project = workspace / "consumer-npm"
    project.mkdir()
    (project / "package.json").write_text(json.dumps({"name": "consumer", "private": True}))
    run(["npm", "install", "--silent", str(tarball), "typescript"], project)

    (project / "esm.mjs").write_text(
        'import { buildSearchSql } from "pghybrid";\n'
        'const { sql } = buildSearchSql({ table: "t", textColumn: "c", vectorColumn: "e" },'
        ' { text: "hi", limit: 3 });\n'
        "if (!sql.includes('SELECT')) { throw new Error('no SQL'); }\n"
        'console.log("esm ok");\n'
    )
    (project / "cjs.cjs").write_text(
        'const { buildSearchSql } = require("pghybrid");\n'
        'const { sql } = buildSearchSql({ table: "t", textColumn: "c", vectorColumn: "e" },'
        ' { text: "hi", limit: 3 });\n'
        "if (!sql.includes('SELECT')) { throw new Error('no SQL'); }\n"
        'console.log("cjs ok");\n'
    )
    esm = run(["node", "esm.mjs"], project, check=False)
    report("ESM import", esm.returncode == 0, esm.stderr.strip()[:80])
    cjs = run(["node", "cjs.cjs"], project, check=False)
    report("CJS require", cjs.returncode == 0, cjs.stderr.strip()[:80])

    # A deliberate type error, so a pass proves the declarations are being read rather
    # than silently resolving to any.
    (project / "consumer.ts").write_text(
        'import { buildSearchSql } from "pghybrid";\n'
        'import type { Config } from "pghybrid";\n'
        'const cfg: Config = { table: "t", textColumn: "c", vectorColumn: "e" };\n'
        'const built = buildSearchSql(cfg, { text: "hi", limit: 3 });\n'
        "const wrong: string = built.params.length;\n"
        "console.log(wrong);\n"
    )
    for resolution, module in (("node16", "node16"), ("bundler", "esnext")):
        (project / "tsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "strict": True,
                        "target": "ES2022",
                        "module": module,
                        "moduleResolution": resolution,
                        "noEmit": True,
                    }
                }
            )
        )
        result = run(["npx", "tsc"], project, check=False)
        caught = "TS2322" in result.stdout
        report(
            f"types resolve under moduleResolution={resolution}",
            caught,
            "declarations resolved to any" if not caught else "",
        )


def main() -> int:
    print("Checking the published artifacts rather than the working tree.")
    with tempfile.TemporaryDirectory() as directory:
        workspace = pathlib.Path(directory)
        try:
            check_python(workspace)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part
            report("python packaging", False, str(exc)[:200])
        if shutil.which("npm"):
            try:
                check_npm(workspace)
            except Exception as exc:  # noqa: BLE001
                report("npm packaging", False, str(exc)[:200])
        else:
            print("\nnpm not on PATH, skipping the tarball checks")

    print()
    if failures:
        print(f"{len(failures)} packaging check(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Both artifacts install and work from empty environments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
