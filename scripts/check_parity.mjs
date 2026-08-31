#!/usr/bin/env node
/**
 * Assert that the Python and TypeScript packages generate the same SQL.
 *
 * Two implementations of one query generator drift apart unless something fails when
 * they do. This script is that something: every fixture below is rendered by both
 * packages and the statements are compared byte for byte, along with the values bound
 * to them.
 *
 * A drift here does not look like a crash. It looks like "the JavaScript results are a
 * bit worse than the Python ones", months later, with no way to tell which half is
 * wrong, so the fixtures deliberately cover the branches that are easy to get subtly
 * different: placeholder numbering, the float formatting inside a pgvector literal,
 * operator precedence in a negated tsquery, and the order values are bound in.
 *
 *     node scripts/check_parity.mjs
 */

import { Buffer } from "node:buffer";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const JS_DIR = join(ROOT, "js");
const PYTHON =
  process.env.PGHYBRID_PYTHON ?? join(ROOT, "python", ".venv", "bin", "python");
const GOLDEN = join(ROOT, "python", "tests", "golden", "canonical_search.sql");

// ---------------------------------------------------------------------------------
// Fixtures
//
// Each one is a config plus one call. They are written once, in the TypeScript
// package's camelCase, and translated to the Python package's field names below, so a
// fixture cannot describe two different queries by accident.
// ---------------------------------------------------------------------------------

const CANONICAL_CONFIG = {
  table: "public.chunks",
  textColumn: "content",
  vectorColumn: "embedding",
  idColumn: "chunk_id",
  tsvectorColumn: "content_tsv",
  language: "english",
  vectorType: "vector",
  metric: "cosine",
  fusion: "rrf",
  k: 60,
  weights: { vector: 1.5, text: 1.0 },
  candidateLimit: 80,
  filterColumns: ["tenant_id", "lang"],
  extraColumns: ["title", "url"],
  recency: { column: "published_at", halfLifeDays: 30.0 },
  paramStyle: "numeric",
  textMatch: "any",
};

const BASE_CONFIG = {
  table: "public.chunks",
  textColumn: "content",
  vectorColumn: "embedding",
  idColumn: "chunk_id",
  tsvectorColumn: "content_tsv",
  filterColumns: ["tenant_id", "lang"],
  extraColumns: ["title", "url"],
};

/** Shorthand for a fixture built on BASE_CONFIG. */
function fixture(name, config, call) {
  return { name, config: { ...BASE_CONFIG, ...config }, call };
}

const FIXTURES = [
  // The golden query from the Python suite: every branch of the builder at once.
  {
    name: "canonical",
    config: CANONICAL_CONFIG,
    call: {
      embedding: [0.25, -0.5, 0.75],
      text: 'renewal "notice period" -pricing',
      limit: 10,
      offset: 20,
      filters: { tenant_id: 42, lang: ["en", "de"] },
      nearMiss: 3,
      highlight: true,
    },
  },

  // The three query shapes. A single-signal statement has to stay interchangeable
  // with a hybrid one, typed NULLs and all.
  fixture("vector-only", {}, { embedding: [0.1, 0.2, 0.3], limit: 5 }),
  fixture("text-only", {}, { text: "renewal notice period", limit: 5 }),
  fixture("hybrid-minimal", {}, { embedding: [0.1], text: "renewal", limit: 5 }),

  // The inline tsvector: no migration, no GIN index, and a language name that is
  // interpolated rather than bound in both packages.
  fixture("inline-tsvector", { tsvectorColumn: null }, { text: "renewal", limit: 5 }),
  fixture(
    "inline-tsvector-german",
    { tsvectorColumn: null, language: "german", queryParser: "plainto_tsquery" },
    { text: "kündigung frist", limit: 5 },
  ),

  // halfvec, where the query vector is cast twice. Both packages emit the redundant
  // cast; if one of them ever stops, this fixture is the thing that notices.
  fixture(
    "halfvec",
    { vectorType: "halfvec", metric: "l2" },
    { embedding: [0.5, 0.25], limit: 5 },
  ),
  fixture(
    "halfvec-hybrid",
    { vectorType: "halfvec", metric: "cosine" },
    { embedding: [0.5, 0.25], text: "renewal notice", limit: 5, highlight: true },
  ),

  // Every metric alias, because the operator is interpolated straight into the
  // statement and a wrong one ranks by the wrong distance without any error.
  ...["cosine", "l2", "euclidean", "inner_product", "ip", "l1", "manhattan"].map((metric) =>
    fixture(`metric-${metric}`, { metric }, { embedding: [0.5, -0.25], limit: 5 }),
  ),

  // Text matching. "all" is one parser call over the raw string; "any" is one call per
  // term, OR-ed. Exclusions are not in the tsquery at all, they are a predicate on both
  // candidate CTEs, so these fixtures pin where they land.
  fixture("text-match-all", { textMatch: "all" }, { text: "renewal notice -pricing", limit: 5 }),
  fixture(
    "text-match-all-phrase",
    { textMatch: "all" },
    { text: 'renewal "notice period"', limit: 5 },
  ),
  fixture("text-match-any-single", { textMatch: "any" }, { text: "renewal", limit: 5 }),
  fixture(
    "text-match-any-negation",
    { textMatch: "any" },
    { text: "renewal notice -pricing", limit: 5 },
  ),
  fixture(
    "text-match-any-two-negations",
    { textMatch: "any" },
    { text: "renewal -pricing -legacy", limit: 5 },
  ),
  fixture(
    "text-match-any-quoted-phrase",
    { textMatch: "any" },
    { text: 'renewal "notice period" clause', limit: 5 },
  ),
  fixture(
    "text-match-any-quoted-negation",
    { textMatch: "any" },
    { text: 'renewal -"legacy plan"', limit: 5 },
  ),
  // Only exclusions, and only noise: neither has a keyword signal, so both are vector
  // search with the exclusion applied. Without an embedding they are rejected instead,
  // which the error fixtures below cover.
  fixture(
    "text-only-exclusions",
    {},
    { embedding: [0.1, 0.2], text: "-pricing -legacy", limit: 5 },
  ),
  fixture("text-noise-only", {}, { embedding: [0.1, 0.2], text: "and or", limit: 5 }),
  // Highlight escaping, on by default and off for non-HTML delimiters.
  fixture("highlight-escaped", {}, { embedding: [0.1], text: "renewal", limit: 5, highlight: true }),
  fixture(
    "highlight-unescaped",
    { escapeHighlight: false, headlineOptions: "StartSel=**, StopSel=**" },
    { embedding: [0.1], text: "renewal", limit: 5, highlight: true },
  ),
  fixture("text-unicode", {}, { text: 'café 日本語 "kündigung frist" -naïve 🙂', limit: 5 }),

  // Filters, inside both candidate CTEs.
  fixture("filter-scalar", {}, { embedding: [0.1], text: "renewal", limit: 5, filters: { tenant_id: 7 } }),
  fixture(
    "filter-array",
    {},
    { embedding: [0.1], text: "renewal", limit: 5, filters: { lang: ["en", "de", "fr"] } },
  ),
  fixture("filter-null", {}, { embedding: [0.1], limit: 5, filters: { tenant_id: null } }),
  fixture("filter-empty-array", {}, { embedding: [0.1], limit: 5, filters: { lang: [] } }),
  fixture(
    "filter-mixed",
    {},
    {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      filters: { tenant_id: "acme-7f3", lang: ["en"] },
    },
  ),

  // Recency decay, which binds one half-life and references it twice.
  fixture(
    "recency",
    { recency: { column: "published_at", halfLifeDays: 30 } },
    { embedding: [0.1], text: "renewal", limit: 5 },
  ),
  fixture(
    "recency-fractional",
    { recency: { column: "created_at", halfLifeDays: 0.5 } },
    { embedding: [0.1], limit: 5 },
  ),

  // Fusion methods and weights.
  fixture("weighted-fusion", { fusion: "weighted" }, { embedding: [0.1], text: "renewal", limit: 5 }),
  fixture(
    "weighted-fusion-override",
    { fusion: "weighted" },
    { embedding: [0.1], text: "renewal", limit: 5, fusion: "rrf" },
  ),
  fixture(
    "custom-weights",
    { weights: { vector: 2.5, text: 0.25 }, k: 0 },
    { embedding: [0.1], text: "renewal", limit: 5 },
  ),
  fixture("zero-text-weight", { weights: { vector: 1, text: 0 } }, { embedding: [0.1], text: "renewal", limit: 5 }),

  // pyformat, where a value referenced twice has to be sent twice.
  fixture(
    "pyformat-hybrid",
    { paramStyle: "pyformat" },
    {
      embedding: [0.1, 0.2],
      text: "renewal notice -pricing",
      limit: 5,
      filters: { tenant_id: 7 },
      highlight: true,
    },
  ),
  fixture("pyformat-vector-only", { paramStyle: "pyformat" }, { embedding: [0.1, 0.2], limit: 5 }),
  fixture(
    "pyformat-recency",
    { paramStyle: "pyformat", recency: { column: "published_at", halfLifeDays: 14 } },
    { embedding: [0.1], text: "renewal", limit: 5 },
  ),

  // Paging, the near-miss band, and the candidate budget it forces upwards.
  fixture("offset-and-near-miss", {}, { embedding: [0.1], limit: 10, offset: 20, nearMiss: 5 }),
  fixture(
    "candidate-limit-raised",
    { candidateLimit: 5 },
    { embedding: [0.1], text: "renewal", limit: 10, nearMiss: 3 },
  ),
  fixture(
    "candidate-limit-override",
    { candidateLimit: 50 },
    { embedding: [0.1], limit: 10, candidateLimit: 120 },
  ),
  fixture("highlight-text-only", {}, { text: "renewal notice", limit: 5, highlight: true }),
  fixture("highlight-ignored", {}, { embedding: [0.1], limit: 5, highlight: true }),
  fixture(
    "custom-headline-options",
    { headlineOptions: "StartSel=[[, StopSel=]], MaxWords=12" },
    { text: "renewal", limit: 5, highlight: true },
  ),
  fixture("ts-rank", { rankFunction: "ts_rank" }, { text: "renewal", limit: 5 }),
  fixture("phraseto", { queryParser: "phraseto_tsquery" }, { text: "renewal notice", limit: 5 }),

  // Float formatting inside the pgvector literal. Python prints 1.0 where JavaScript
  // prints 1, and the two languages disagree again about when to switch to exponent
  // notation, so the vector literal is the single most likely place for the two ports
  // to diverge silently.
  fixture("vector-integral", {}, { embedding: [1, 0, -1, 2], limit: 5 }),
  fixture("vector-empty", {}, { embedding: [], limit: 5 }),
  fixture(
    "vector-extremes",
    {},
    { embedding: [1e16, 1e-5, 1e15, 0.0001, 1.5e-7, -0.0, 123456789.123456789], limit: 5 },
  ),
  fixture("vector-precision", {}, { embedding: [0.1 + 0.2, 1 / 3, Math.PI, 2 ** 53], limit: 5 }),
];

// ---------------------------------------------------------------------------------
// Translating a fixture into the Python package's vocabulary
//
// Explicit maps rather than a camelCase-to-snake_case function: `paramStyle` is
// `paramstyle` in Python, and a silent mistranslation would have the two packages
// comparing two different queries and agreeing about it.
// ---------------------------------------------------------------------------------

const CONFIG_KEYS = {
  table: "table",
  textColumn: "text_column",
  vectorColumn: "vector_column",
  idColumn: "id_column",
  tsvectorColumn: "tsvector_column",
  language: "language",
  vectorType: "vector_type",
  metric: "metric",
  fusion: "fusion",
  k: "k",
  weights: "weights",
  candidateLimit: "candidate_limit",
  filterColumns: "filter_columns",
  extraColumns: "extra_columns",
  recency: "recency",
  queryParser: "query_parser",
  rankFunction: "rank_function",
  paramStyle: "paramstyle",
  textMatch: "text_match",
  headlineOptions: "headline_options",
  escapeHighlight: "escape_highlight",
};

const CALL_KEYS = {
  embedding: "embedding",
  text: "text",
  limit: "limit",
  offset: "offset",
  filters: "filters",
  candidateLimit: "candidate_limit",
  nearMiss: "near_miss",
  highlight: "highlight",
  fusion: "fusion",
};

const RECENCY_KEYS = { column: "column", halfLifeDays: "half_life_days" };

function rename(object, keys, what) {
  const out = {};
  for (const [key, value] of Object.entries(object)) {
    const renamed = keys[key];
    if (renamed === undefined) {
      throw new Error(`fixture has no ${what} field named ${key}`);
    }
    out[renamed] = value;
  }
  return out;
}

function toPythonFixture({ name, config, call }) {
  const pythonConfig = rename(config, CONFIG_KEYS, "config");
  if (pythonConfig.recency) {
    pythonConfig.recency = rename(pythonConfig.recency, RECENCY_KEYS, "recency");
  }
  // embedding and text have no defaults in the Python builder, so both are always sent.
  const pythonCall = { embedding: null, text: null, ...rename(call, CALL_KEYS, "call") };
  if (pythonCall.embedding) {
    pythonCall.embedding = pythonCall.embedding.map(forJson);
  }
  return { name, config: pythonConfig, call: pythonCall };
}

/**
 * JSON.stringify writes -0 as 0, so a negative zero in an embedding would reach Python
 * as a positive one and the two packages would look like they disagree about a value
 * neither of them got wrong. Python's float() reads the string back as -0.0, and the
 * builder calls float() on every coordinate anyway.
 */
function forJson(value) {
  return Object.is(value, -0) ? "-0.0" : value;
}

// ---------------------------------------------------------------------------------
// Rendering both sides
// ---------------------------------------------------------------------------------

const PYTHON_DRIVER = `
import json
import sys
from pathlib import Path

# The working tree, not whatever copy happens to be installed: this check is about the
# code in this repository.
sys.path.insert(0, str(Path(${JSON.stringify(ROOT)}) / "python" / "src"))

from pghybrid.config import Config
from pghybrid.sql import build_search_sql

payload = json.load(sys.stdin)

rendered = []
for fixture in payload["fixtures"]:
    sql, params = build_search_sql(Config(**fixture["config"]), **fixture["call"])
    rendered.append({"name": fixture["name"], "sql": sql, "params": params})

# What the two packages refuse is as much a part of the contract as what they emit. A
# statement one of them builds and the other rejects is the worst kind of divergence,
# because it only shows up in whichever language the user happened to pick.
errors = []
for fixture in payload["errors"]:
    try:
        build_search_sql(Config(**fixture["config"]), **fixture["call"])
    except Exception as exc:
        errors.append({"name": fixture["name"], "message": str(exc)})
    else:
        errors.append({"name": fixture["name"], "message": None})

json.dump({"rendered": rendered, "errors": errors}, sys.stdout)
`;

/**
 * Compare two error messages, allowing only the naming convention to differ.
 *
 * A message that names the option to change is far more useful than one that does not,
 * and the option is `candidate_limit` in Python and `candidateLimit` in TypeScript. That
 * difference is deliberate; every other difference is a bug. So both sides are folded to
 * snake_case before comparison, which leaves any real divergence in wording, values or
 * punctuation visible.
 */
function sameMessage(expected, actual) {
  if (expected === null || actual === null) {
    return expected === actual;
  }
  const fold = (text) => text.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
  return fold(expected) === fold(actual);
}

/**
 * Inputs both packages must refuse, with the same sentence.
 *
 * Rejection is part of the contract. A query one language builds a statement for and the
 * other throws on is the worst kind of divergence: it is invisible until someone ports
 * their code, and the SQL fixtures above can never catch it, because there is no SQL to
 * compare.
 *
 * The messages are compared verbatim rather than by class. They are the whole of what a
 * user sees, and two packages that disagree about how to explain the same mistake are
 * two different libraries.
 */
const ERROR_FIXTURES = [
  fixture("error-no-signal-at-all", {}, { limit: 5 }),
  // Exclusions with nothing to rank: there is no keyword signal to fall back on, and
  // inverting the corpus would be worse than saying so.
  fixture("error-only-exclusions", {}, { text: "-pricing", limit: 5 }),
  fixture("error-only-exclusions-quoted", {}, { text: '-"legacy plan" -pricing', limit: 5 }),
  fixture("error-noise-only", {}, { text: "and or", limit: 5 }),
  fixture("error-blank-query", {}, { text: "   ", limit: 5 }),
  fixture("error-limit-zero", {}, { embedding: [0.1], limit: 0 }),
  fixture("error-limit-negative", {}, { embedding: [0.1], limit: -1 }),
  fixture("error-negative-offset", {}, { embedding: [0.1], limit: 5, offset: -1 }),
  // A page outside the candidate pool. Widening the pool per page would reorder every
  // page, so this is an error rather than an empty result.
  fixture("error-page-past-the-pool", {}, { embedding: [0.1], limit: 10, offset: 50 }),
  // A table column that shadows one the statement already returns. Postgres allows the
  // duplicate name and the driver keeps the last one, so the computed value vanishes
  // without an error anywhere.
  fixture("error-extra-column-shadows-text-rank", { extraColumns: ["text_rank"] },
    { embedding: [0.1], limit: 5 }),
  fixture("error-extra-columns-shadow-two", { extraColumns: ["score", "fused_score", "title"] },
    { embedding: [0.1], limit: 5 }),
  fixture("error-text-column-shadows-highlight", { textColumn: "highlight" },
    { embedding: [0.1], limit: 5 }),
  fixture(
    "error-page-past-an-explicit-pool",
    {},
    { embedding: [0.1], limit: 10, offset: 30, candidateLimit: 25 },
  ),
];

function renderWithPython(fixtures) {
  if (!existsSync(PYTHON)) {
    throw new Error(
      `no Python interpreter at ${PYTHON}. Create one with ` +
        "`cd python && uv venv && uv pip install -e \".[dev]\"`, or point " +
        "PGHYBRID_PYTHON at one.",
    );
  }
  const result = spawnSync(PYTHON, ["-c", PYTHON_DRIVER], {
    input: JSON.stringify({
      fixtures: fixtures.map(toPythonFixture),
      errors: ERROR_FIXTURES.map(toPythonFixture),
    }),
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(`the Python builder failed:\n${result.stderr || result.stdout}`);
  }
  return JSON.parse(result.stdout);
}

/**
 * Build the TypeScript package if the compiled output is missing or stale.
 *
 * The parity job installs dependencies but does not build, and importing a stale dist
 * would compare the previous commit's TypeScript against this commit's Python, a
 * green check that means nothing.
 */
function ensureBuilt() {
  const dist = join(JS_DIR, "dist", "index.js");
  const newestSource = newestMtime(join(JS_DIR, "src"));
  if (existsSync(dist) && statSync(dist).mtimeMs >= newestSource) {
    return;
  }
  process.stderr.write("building the TypeScript package...\n");
  execFileSync("npm", ["run", "--silent", "build"], { cwd: JS_DIR, stdio: "inherit" });
}

function newestMtime(directory) {
  let newest = 0;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    const mtime = entry.isDirectory() ? newestMtime(path) : statSync(path).mtimeMs;
    newest = Math.max(newest, mtime);
  }
  return newest;
}

// ---------------------------------------------------------------------------------
// Comparing
// ---------------------------------------------------------------------------------

/**
 * Compare two bound values.
 *
 * Numbers are compared numerically because JavaScript has one number type: Python's
 * 60.0 and JavaScript's 60 are the same value and bind identically. Everything else
 * has to match structurally, including the order of an array bound to `= ANY($n)`.
 */
function sameParams(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** The first line that differs, with a little context on either side. */
function firstDifference(expected, actual) {
  const expectedLines = expected.split("\n");
  const actualLines = actual.split("\n");
  const count = Math.max(expectedLines.length, actualLines.length);
  for (let index = 0; index < count; index += 1) {
    if (expectedLines[index] !== actualLines[index]) {
      const from = Math.max(0, index - 2);
      const context = expectedLines
        .slice(from, index)
        .map((line, offset) => `      ${String(from + offset + 1).padStart(3)} | ${line}`);
      return [
        ...context,
        `  py  ${String(index + 1).padStart(3)} | ${expectedLines[index] ?? "<end of statement>"}`,
        `  ts  ${String(index + 1).padStart(3)} | ${actualLines[index] ?? "<end of statement>"}`,
      ].join("\n");
    }
  }
  return "  the statements differ in bytes but not in lines (line endings?)";
}

async function main() {
  ensureBuilt();
  const { buildSearchSql } = await import(new URL("../js/dist/index.js", import.meta.url).href);

  const { rendered: python, errors: pythonErrors } = renderWithPython(FIXTURES);
  const failures = [];

  for (const [index, item] of FIXTURES.entries()) {
    const expected = python[index];
    if (expected === undefined || expected.name !== item.name) {
      throw new Error(`the Python builder returned no result for ${item.name}`);
    }

    const actual = buildSearchSql(item.config, {
      embedding: null,
      text: null,
      ...item.call,
    });

    const sqlMatches =
      Buffer.compare(Buffer.from(expected.sql, "utf8"), Buffer.from(actual.sql, "utf8")) === 0;
    const paramsMatch = sameParams(expected.params, actual.params);

    if (sqlMatches && paramsMatch) {
      process.stdout.write(`  ok   ${item.name}\n`);
      continue;
    }

    process.stdout.write(`  FAIL ${item.name}\n`);
    const report = [`${item.name}:`];
    if (!sqlMatches) {
      report.push("  the statements differ:");
      report.push(firstDifference(expected.sql, actual.sql));
    }
    if (!paramsMatch) {
      report.push("  the bound parameters differ:");
      report.push(`  py  ${JSON.stringify(expected.params)}`);
      report.push(`  ts  ${JSON.stringify(actual.params)}`);
    }
    failures.push(report.join("\n"));
  }

  // Rejections. Both packages must refuse the same inputs, with the same sentence.
  for (const [index, item] of ERROR_FIXTURES.entries()) {
    const expected = pythonErrors[index];
    if (expected === undefined || expected.name !== item.name) {
      throw new Error(`the Python builder returned no result for ${item.name}`);
    }

    let actual = null;
    try {
      buildSearchSql(item.config, { embedding: null, text: null, ...item.call });
    } catch (error) {
      actual = error instanceof Error ? error.message : String(error);
    }

    if (sameMessage(expected.message, actual)) {
      process.stdout.write(`  ok   ${item.name}\n`);
      continue;
    }

    process.stdout.write(`  FAIL ${item.name}\n`);
    failures.push(
      [
        `${item.name}:`,
        expected.message === null || actual === null
          ? "  one package builds a statement and the other refuses:"
          : "  both refuse, but say different things:",
        `  py  ${expected.message === null ? "<built a statement>" : JSON.stringify(expected.message)}`,
        `  ts  ${actual === null ? "<built a statement>" : JSON.stringify(actual)}`,
      ].join("\n"),
    );
  }

  // The golden file is the Python suite's own snapshot. Checking the TypeScript output
  // against it as well means the check still catches a drift if both builders are
  // changed together and only the snapshot is left behind.
  const canonical = FIXTURES[0];
  const golden = readFileSync(GOLDEN, "utf8");
  const built = buildSearchSql(canonical.config, {
    embedding: null,
    text: null,
    ...canonical.call,
  });
  if (built.sql + "\n" !== golden) {
    failures.push(
      [
        "golden snapshot:",
        "  TypeScript does not reproduce the committed snapshot.",
        "  (Below, `py` is the committed snapshot and `ts` is what the builder produced.",
        "   A difference here usually means the snapshot is stale, not that the two",
        "   packages disagree: regenerate with PGHYBRID_UPDATE_GOLDEN=1.)",
        firstDifference(golden, built.sql + "\n"),
      ].join("\n"),
    );
    process.stdout.write("  FAIL golden snapshot\n");
  } else {
    process.stdout.write("  ok   golden snapshot\n");
  }

  if (failures.length > 0) {
    process.stderr.write(`\n${failures.join("\n\n")}\n\n`);
    process.stderr.write(
      `${failures.length} of ${FIXTURES.length + ERROR_FIXTURES.length + 1} checks ` +
        "failed. The Python and " +
        "TypeScript packages no longer generate the same SQL; fix whichever one moved " +
        "before this is released as one library with two behaviours.\n",
    );
    process.exitCode = 1;
    return;
  }

  process.stdout.write(
    `\n${FIXTURES.length} fixtures, ${ERROR_FIXTURES.length} rejections and the golden ` +
      "snapshot: identical SQL, parameters and error messages.\n",
  );
}

await main();
