#!/usr/bin/env bash
# Run locally exactly what CI runs, in the same order, and stop at the first failure.
#
# This exists because a change once passed `npm test` and `npm run lint` locally and then
# failed CI on `npm run typecheck` — vitest does not typecheck, so a type error in a test
# file is invisible to the test run. Any gap between "what I ran" and "what CI runs" ends
# up being discovered by CI, which is slower and noisier than discovering it here.
#
#   scripts/check_all.sh          # everything
#   scripts/check_all.sh fast     # skip the packaging build (the slow one)
#
# Needs the test database: docker compose up -d
set -euo pipefail
cd "$(dirname "$0")/.."

export PGHYBRID_TEST_DSN="${PGHYBRID_TEST_DSN:-postgresql://postgres:pghybrid@localhost:55432/pghybrid}"
export PGHYBRID_DSN="${PGHYBRID_DSN:-$PGHYBRID_TEST_DSN}"

step() { printf '\n\033[1m▸ %s\033[0m\n' "$1"; }

step "python · ruff check";        python/.venv/bin/ruff check python/src python/tests
step "python · ruff format";       python/.venv/bin/ruff format --check python/src python/tests
step "python · mypy --strict";     (cd python && .venv/bin/mypy src)
step "python · pytest";            (cd python && .venv/bin/python -m pytest -q)

step "typescript · eslint";        (cd js && npm run --silent lint)
step "typescript · tsc --noEmit";  (cd js && npm run --silent typecheck)
step "typescript · vitest";        (cd js && npm test --silent)

step "parity · identical SQL";     node scripts/check_parity.mjs
step "sql/ · runs unmodified";     python/.venv/bin/python scripts/check_standalone_sql.py
step "README · output matches";    python/.venv/bin/python scripts/check_readme_output.py
step "demo image · matches";       python/.venv/bin/python scripts/make_demo_svg.py --check
step "examples · still run";       python/.venv/bin/python examples/three_way_comparison.py >/dev/null
                                   python/.venv/bin/python examples/langchain_retriever.py >/dev/null

if [ "${1:-}" != "fast" ]; then
  step "packaging · as a stranger";  python/.venv/bin/python scripts/check_packaging.py
fi

printf '\n\033[32mEverything CI runs passes locally.\033[0m\n'
