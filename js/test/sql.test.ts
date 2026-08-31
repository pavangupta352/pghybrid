/**
 * Unit tests for the SQL builder.
 *
 * Every test here is a pure function of a config and some arguments: no connection, no
 * fixtures with side effects, no network. That is the whole point of keeping SQL
 * generation separate from execution — the statement can be asserted on directly, and
 * a regression in the query shape is caught before anyone has to notice bad search
 * results.
 *
 * The assertions favour naming the failure mode over matching the text. A test called
 * "fuses with a full outer join" that fails tells you the library just started
 * returning the intersection of the two signals; a test called "sql shape" tells you
 * nothing.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import type { Config } from "../src/config.js";
import { opsClass } from "../src/config.js";
import { buildSearchSql, IdentifierError, Params, quoteIdent } from "../src/sql.js";

/** The Python package's snapshot, which this port has to reproduce byte for byte. */
const GOLDEN = fileURLToPath(
  new URL("../../python/tests/golden/canonical_search.sql", import.meta.url),
);

function makeConfig(overrides: Partial<Config> = {}): Config {
  // Deliberately richer than the library's own defaults: a schema-qualified table, a
  // non-default id column, a stored tsvector, filterable columns and passthrough
  // columns each switch on a branch of the builder that a bare three-field config
  // would leave untested.
  return {
    table: "public.chunks",
    textColumn: "content",
    vectorColumn: "embedding",
    idColumn: "chunk_id",
    tsvectorColumn: "content_tsv",
    filterColumns: ["tenant_id", "lang"],
    extraColumns: ["title", "url"],
    ...overrides,
  };
}

const config = makeConfig();

// ---------------------------------------------------------------------------------
// Helpers for reading the generated statement.
//
// The builder emits every CTE-internal SELECT indented and the final SELECT at column
// zero, which is what lets these split the statement without a SQL parser.
// ---------------------------------------------------------------------------------

/** The body of one named CTE. */
function cte(sql: string, name: string): string {
  const match = new RegExp(`${name} AS \\(([\\s\\S]*?)\\n\\)`).exec(sql);
  expect(match, `no ${name} CTE in:\n${sql}`).not.toBeNull();
  return match![1]!;
}

/** Everything before the final SELECT: the WITH clause and nothing else. */
function cteBlock(sql: string): string {
  const index = sql.indexOf("\nSELECT ");
  expect(index, `no top-level SELECT in:\n${sql}`).toBeGreaterThan(-1);
  return sql.slice(0, index);
}

/** The final SELECT onwards, with the CTEs stripped off. */
function finalSelect(sql: string): string {
  const index = sql.indexOf("\nSELECT ");
  expect(index, `no top-level SELECT in:\n${sql}`).toBeGreaterThan(-1);
  return sql.slice(index + 1);
}

/** The column names a caller actually receives, in order. */
function outputAliases(sql: string): string[] {
  const selectList = finalSelect(sql).split("\nFROM fused f")[0]!.slice("SELECT ".length);
  return selectList.split(",\n       ").map((column) => {
    const trimmed = column.trim();
    const alias = trimmed.includes(" AS ")
      ? trimmed.slice(trimmed.lastIndexOf(" AS ") + 4)
      : trimmed.slice(trimmed.lastIndexOf(".") + 1);
    return alias.trim().replace(/^"|"$/g, "");
  });
}

/** The value bound to the LIMIT in a fragment. Numeric paramstyle only. */
function boundLimit(fragment: string, params: unknown[]): unknown {
  const match = /LIMIT \$(\d+)/.exec(fragment);
  expect(match, `no LIMIT in:\n${fragment}`).not.toBeNull();
  return params[Number(match![1]) - 1];
}

function count(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

// ---------------------------------------------------------------------------------
// quoteIdent
// ---------------------------------------------------------------------------------

describe("quoteIdent", () => {
  it.each([
    ["chunks", '"chunks"'],
    ["public.chunks", '"public"."chunks"'],
    ["_internal", '"_internal"'],
    ["col$1", '"col$1"'],
    ["MixedCase", '"MixedCase"'],
    ["analytics.Events2024", '"analytics"."Events2024"'],
  ])("quotes %s unconditionally", (name, expected) => {
    // Quoting is unconditional so a name that collides with a keyword still works.
    expect(quoteIdent(name)).toBe(expected);
  });

  it.each([
    'a"; DROP TABLE users; --',
    "chunks; DELETE FROM chunks",
    "chunks'",
    "content, (SELECT secret FROM keys)",
    "tab-le",
    "1abc",
    "two words",
    "public.chunks; --",
    "  ",
  ])("refuses the injection attempt %j", (name) => {
    // Names outside the identifier alphabet are refused, not escaped. Escaping would
    // work, but a column name that needs escaping is far more likely to be an injected
    // string than a deliberate choice, so the failure is loud.
    expect(() => quoteIdent(name)).toThrow(IdentifierError);
  });

  it.each(["", null, 0, [], "a."])("refuses the empty or missing name %j", (name) => {
    expect(() => quoteIdent(name as never)).toThrow(IdentifierError);
  });

  it.each(["a.b.c", "db.public.chunks", "w.x.y.z"])(
    "refuses the three-part name %s",
    (name) => {
      // Postgres has no database-qualified references in a query; catch the confusion.
      expect(() => quoteIdent(name)).toThrow(/schema\.name/);
    },
  );

  it("fails at build time for a bad name in the config", () => {
    // A bad name in the config fails when the SQL is built, not when it is executed.
    expect(() =>
      buildSearchSql(makeConfig({ table: "chunks; DROP TABLE users" }), {
        embedding: [0.1],
        limit: 5,
      }),
    ).toThrow(IdentifierError);
  });
});

// ---------------------------------------------------------------------------------
// Placeholder rendering
// ---------------------------------------------------------------------------------

describe("placeholder rendering", () => {
  it("reuses one numeric placeholder for a repeated value", () => {
    // The distance expression appears three times in the vector CTE (projection,
    // window ORDER BY, and the CTE's own ORDER BY). Numbered placeholders let all
    // three point at a single copy of the embedding, which matters: an embedding is
    // the largest value in the statement and sending it three times is measurable on
    // the wire.
    const { sql, params } = buildSearchSql(config, { embedding: [0.1, 0.2, 0.3], limit: 5 });
    expect(count(cte(sql, "vector_candidates"), "$1::vector")).toBe(3);
    expect(params.filter((value) => value === "[0.1,0.2,0.3]")).toHaveLength(1);
  });

  it("repeats the value for every mention in pyformat", () => {
    // %s is positional, so a value used twice has to be sent twice.
    const { sql, params } = buildSearchSql(makeConfig({ paramStyle: "pyformat" }), {
      embedding: [0.1, 0.2, 0.3],
      limit: 5,
    });
    expect(count(cte(sql, "vector_candidates"), "%s::vector")).toBe(3);
    expect(params.filter((value) => value === "[0.1,0.2,0.3]")).toHaveLength(3);
    expect(sql).not.toContain("$1");
  });

  it("produces a different parameter count for each style", () => {
    // This is the property the whole Params indirection exists for. If both styles
    // ever produce the same count, one of them is wrong: either numeric stopped
    // deduplicating repeated references, or pyformat stopped repeating them and the
    // driver is about to receive fewer values than the statement has placeholders.
    const options = {
      embedding: [0.1, 0.2, 0.3],
      text: "renewal notice",
      limit: 5,
      filters: { tenant_id: 7 },
      highlight: true,
    };
    const numeric = buildSearchSql(makeConfig(), options);
    const pyformat = buildSearchSql(makeConfig({ paramStyle: "pyformat" }), options);

    expect(pyformat.params.length).toBeGreaterThan(numeric.params.length);
    expect(numeric.params).toHaveLength(new Set(numeric.sql.match(/\$\d+/g)).size);
    expect(pyformat.params).toHaveLength(count(pyformat.sql, "%s"));
    // Same statement, different placeholder syntax: normalising one to the other has
    // to produce the same skeleton.
    expect(numeric.sql.replace(/\$\d+/g, "?")).toBe(pyformat.sql.replaceAll("%s", "?"));
  });

  it("escapes a literal percent for pyformat", () => {
    // A bare % is read as the start of a placeholder by psycopg, so it is doubled. No
    // current code path emits a literal percent, but the renderer is the only place
    // that could, and the escaping has to survive the next feature that needs a LIKE
    // pattern or a modulo.
    const params = new Params();
    const slot = params.add("acme");
    const { sql, params: values } = params.render(
      `SELECT ${slot} WHERE tenant LIKE 'a%b'`,
      "pyformat",
    );
    expect(sql).toBe("SELECT %s WHERE tenant LIKE 'a%%b'");
    expect(values).toEqual(["acme"]);
  });

  it("leaves a literal percent alone for numeric", () => {
    // Doubling the percent for a driver that does not use pyformat would corrupt it.
    const params = new Params();
    const slot = params.add("acme");
    const { sql, params: values } = params.render(
      `SELECT ${slot} WHERE tenant LIKE 'a%b'`,
      "numeric",
    );
    expect(sql).toBe("SELECT $1 WHERE tenant LIKE 'a%b'");
    expect(values).toEqual(["acme"]);
  });

  it("rejects an unknown paramStyle in the renderer", () => {
    // The config catches this earlier; the renderer is the backstop for direct callers.
    expect(() => new Params().render("SELECT 1", "qmark" as never)).toThrow(
      /numeric.*pyformat/s,
    );
  });

  it("never interpolates a caller's value into the statement", () => {
    const { sql, params } = buildSearchSql(makeConfig(), {
      embedding: [0.123456789],
      text: "quarterly renewal",
      limit: 5,
      filters: { tenant_id: "acme-tenant-7f3" },
      highlight: true,
    });
    for (const secret of ["0.123456789", "quarterly", "renewal", "acme-tenant-7f3"]) {
      expect(sql).not.toContain(secret);
    }
    expect(params).toContain("acme-tenant-7f3");
  });
});

// ---------------------------------------------------------------------------------
// The three query shapes
// ---------------------------------------------------------------------------------

describe("the three query shapes", () => {
  it("builds a distinct statement for each signal combination", () => {
    const vectorOnly = buildSearchSql(config, { embedding: [0.1], limit: 5 }).sql;
    const textOnly = buildSearchSql(config, { text: "renewal", limit: 5 }).sql;
    const hybrid = buildSearchSql(config, { embedding: [0.1], text: "renewal", limit: 5 }).sql;

    expect(new Set([vectorOnly, textOnly, hybrid]).size).toBe(3);

    expect(vectorOnly).toContain("vector_candidates AS (");
    expect(vectorOnly).not.toContain("text_candidates AS (");

    expect(textOnly).toContain("text_candidates AS (");
    expect(textOnly).not.toContain("vector_candidates AS (");

    expect(hybrid).toContain("vector_candidates AS (");
    expect(hybrid).toContain("text_candidates AS (");
  });

  it("exposes the same output columns from all three shapes", () => {
    // A single-signal search returns the same row shape as a hybrid one, so the two
    // can be compared side by side. That comparison is only honest if the three
    // statements are interchangeable to the caller, so the columns for the missing
    // signal are selected as typed NULLs rather than dropped.
    const expected = [
      "id",
      "score",
      "fused_score",
      "vector_rank",
      "vector_distance",
      "vector_contribution",
      "text_rank",
      "text_score",
      "text_contribution",
      "content",
      "title",
      "url",
    ];
    const shapes = [
      { embedding: [0.1], limit: 5 },
      { text: "renewal", limit: 5 },
      { embedding: [0.1], text: "renewal", limit: 5 },
    ];
    for (const shape of shapes) {
      expect(outputAliases(buildSearchSql(config, shape).sql)).toEqual(expected);
    }
  });

  it("types the missing signal's NULLs", () => {
    // An untyped NULL makes Postgres guess the column type, which breaks drivers.
    const vectorOnly = buildSearchSql(config, { embedding: [0.1], limit: 5 }).sql;
    expect(vectorOnly).toContain("NULL::bigint AS text_rank");
    expect(vectorOnly).toContain("NULL::double precision AS text_score");

    const textOnly = buildSearchSql(config, { text: "renewal", limit: 5 }).sql;
    expect(textOnly).toContain("NULL::bigint AS vector_rank");
    expect(textOnly).toContain("NULL::double precision AS vector_distance");
  });

  it("deduplicates passthrough columns", () => {
    // Naming the text column again in extraColumns must not select it twice.
    const { sql } = buildSearchSql(makeConfig({ extraColumns: ["content", "title", "title"] }), {
      embedding: [0.1],
      limit: 5,
    });
    const aliases = outputAliases(sql);
    expect(aliases.filter((alias) => alias === "content")).toHaveLength(1);
    expect(aliases.filter((alias) => alias === "title")).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------------

describe("fusion", () => {
  it("fuses with a full outer join, never an inner join", () => {
    // That is the single most damaging regression this file can catch: the query still
    // runs, still returns rows, and still looks plausible — it just quietly drops every
    // document that only one of the two signals found, which is most of the ones
    // hybrid search exists to surface.
    const { sql } = buildSearchSql(config, { embedding: [0.1], text: "renewal", limit: 5 });
    const scored = cte(sql, "scored");
    expect(scored).toContain("FULL OUTER JOIN text_candidates t ON v.id = t.id");
    expect(scored.match(/(?:\w+ )*JOIN/g)).toEqual(["FULL OUTER JOIN"]);
    // A row found by one signal has no rank in the other, so its missing contribution
    // has to fall back to zero rather than NULL, which would null out the whole sum.
    expect(count(scored, "coalesce(")).toBe(3);
  });

  it("computes the RRF contribution as weight over k plus rank", () => {
    const { sql, params } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
    });
    const scored = cte(sql, "scored");
    expect(scored).toMatch(/coalesce\(\$\d+::float8 \/ \(\$\d+::float8 \+ v\.rank\), 0\) AS vector_contribution/);
    expect(scored).toMatch(/coalesce\(\$\d+::float8 \/ \(\$\d+::float8 \+ t\.rank\), 0\) AS text_contribution/);
    // k is bound once and referenced by both contributions.
    expect(params.filter((value) => value === 60)).toHaveLength(1);
  });

  it("scores weighted fusion on the raw signals", () => {
    // Kept because people ask for it, and asserted so its trap stays visible.
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      fusion: "weighted",
    });
    const scored = cte(sql, "scored");
    expect(scored).toMatch(/coalesce\(\$\d+::float8 \* \(1\.0 - v\.distance\), 0\) AS vector_contribution/);
    expect(scored).toMatch(/coalesce\(\$\d+::float8 \* t\.score, 0\) AS text_contribution/);
    // Cosine distance is bounded and ts_rank is not, so the nominal weights do not
    // describe the actual influence of each signal. Nothing here should quietly start
    // normalising the two scales.
    expect(scored).not.toContain("rank)");
  });

  it("lets the fusion argument override the config", () => {
    const cfg = makeConfig({ fusion: "weighted" });
    const byDefault = buildSearchSql(cfg, { embedding: [0.1], text: "renewal", limit: 5 }).sql;
    const overridden = buildSearchSql(cfg, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      fusion: "rrf",
    }).sql;
    expect(byDefault).toContain("1.0 - v.distance");
    expect(overridden).not.toContain("1.0 - v.distance");
  });

  it("binds the weights instead of interpolating them", () => {
    // Tuning weights must not require regenerating the SQL, or the plan cache is lost.
    const { params } = buildSearchSql(makeConfig({ weights: { vector: 2.5, text: 0.25 } }), {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
    });
    expect(params).toContain(2.5);
    expect(params).toContain(0.25);
  });

  it("rejects an unknown fusion method", () => {
    expect(() =>
      buildSearchSql(config, { embedding: [0.1], limit: 5, fusion: "borda" as never }),
    ).toThrow(/borda.*rrf.*weighted/s);
  });
});

// ---------------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------------

describe("filters", () => {
  it("applies filters inside both candidate CTEs", () => {
    // Filtering after the fusion is the classic way to destroy recall: each signal
    // spends its candidate budget on rows the caller has already excluded, and a
    // tenant with few documents gets an empty result set from a query that matches
    // plenty of its rows.
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      filters: { tenant_id: 7 },
    });
    expect(count(sql, '"tenant_id" = ')).toBe(2);
    expect(cte(sql, "vector_candidates")).toContain('"tenant_id" = ');
    expect(cte(sql, "text_candidates")).toContain('"tenant_id" = ');
  });

  it("does not repeat the filter after the fusion", () => {
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      filters: { tenant_id: 7 },
    });
    expect(cte(sql, "scored")).not.toContain("tenant_id");
    expect(cte(sql, "fused")).not.toContain("tenant_id");
    expect(finalSelect(sql)).not.toContain("tenant_id");
  });

  it("applies the filter once for a single-signal query", () => {
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      limit: 5,
      filters: { tenant_id: 7 },
    });
    expect(count(sql, '"tenant_id" = ')).toBe(1);
  });

  it("ANDs filters onto the existing WHERE clause", () => {
    // The vector CTE already excludes NULL embeddings; a filter extends that clause.
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      limit: 5,
      filters: { tenant_id: 7, lang: "en" },
    });
    expect(cte(sql, "vector_candidates")).toMatch(
      /WHERE "embedding" IS NOT NULL AND "tenant_id" = \$\d+ AND "lang" = \$\d+/,
    );
  });

  it("rejects an undeclared filter column", () => {
    // Filter columns are declared up front so they can be validated and indexed.
    expect(() =>
      buildSearchSql(config, { embedding: [0.1], limit: 5, filters: { deleted_at: null } }),
    ).toThrow(/deleted_at.*tenant_id.*lang/s);
  });

  it("rejects filters when no filter columns are declared", () => {
    expect(() =>
      buildSearchSql(makeConfig({ filterColumns: [] }), {
        embedding: [0.1],
        limit: 5,
        filters: { tenant_id: 7 },
      }),
    ).toThrow(/filterColumns/);
  });

  it("turns an empty list filter into FALSE rather than invalid SQL", () => {
    // IN () is a syntax error in Postgres; an empty set means no rows, so say so.
    const { sql, params } = buildSearchSql(config, {
      embedding: [0.1],
      limit: 5,
      filters: { lang: [] },
    });
    expect(sql).toContain('"embedding" IS NOT NULL AND FALSE');
    expect(sql).not.toContain("IN ()");
    expect(sql).not.toContain("ANY()");
    expect(params.some((value) => Array.isArray(value))).toBe(false);
  });

  it("turns a null filter into IS NULL", () => {
    // = NULL is never true; the caller meant IS NULL.
    const { sql, params } = buildSearchSql(config, {
      embedding: [0.1],
      limit: 5,
      filters: { tenant_id: null },
    });
    expect(sql).toContain('"tenant_id" IS NULL');
    expect(params).not.toContain(null);
  });

  it.each([
    ["array", ["en", "de"]],
    ["set", new Set(["en"])],
  ])("binds one array parameter for a %s filter", (_kind, value) => {
    // = ANY($n) keeps the placeholder count independent of the number of values.
    // Expanding to IN ($1, $2, $3) would generate a different statement for every list
    // length and defeat the server's prepared-statement cache.
    const { sql, params } = buildSearchSql(config, {
      embedding: [0.1],
      limit: 5,
      filters: { lang: value },
    });
    expect(sql).toMatch(/"lang" = ANY\(\$\d+\)/);
    expect(params.some((param) => Array.isArray(param))).toBe(true);
  });
});

// ---------------------------------------------------------------------------------
// Text side
// ---------------------------------------------------------------------------------

describe("the text side", () => {
  it("uses the stored tsvector column when configured", () => {
    const { sql } = buildSearchSql(config, { text: "renewal", limit: 5 });
    expect(sql).toContain('"content_tsv" @@ tsq');
    expect(sql).not.toContain("to_tsvector(");
  });

  it("uses the two-argument form for an inline tsvector", () => {
    // to_tsvector(text) is STABLE, not IMMUTABLE: it reads default_text_search_config.
    // The one-argument form cannot be indexed, and a session that sets a different
    // search configuration silently changes what the query means.
    const { sql } = buildSearchSql(makeConfig({ tsvectorColumn: null }), {
      text: "renewal",
      limit: 5,
    });
    expect(sql).toContain("to_tsvector('english', coalesce(\"content\", ''))");
  });

  it("evaluates ts_headline only after ranking", () => {
    // ts_headline re-parses the document text; running it per candidate is ruinous.
    // Inside a candidate CTE it would run for every row the signal considered. In the
    // final SELECT it runs only for the page being returned.
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      highlight: true,
    });
    expect(count(sql, "ts_headline(")).toBe(1);
    expect(finalSelect(sql)).toContain("ts_headline(");
    expect(cteBlock(sql)).not.toContain("ts_headline(");
    expect(outputAliases(sql)).toContain("highlight");
  });

  it("reuses the parsed tsquery for the headline", () => {
    // Re-parsing the query string for the headline could highlight different terms.
    const { sql } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      highlight: true,
    });
    expect(finalSelect(sql)).toContain("(SELECT tsq FROM text_query)");
  });

  it("ignores highlight without a text signal", () => {
    // There is no tsquery to highlight against, so asking for one is a no-op.
    const { sql } = buildSearchSql(config, { embedding: [0.1], limit: 5, highlight: true });
    expect(sql).not.toContain("ts_headline");
    expect(outputAliases(sql)).not.toContain("highlight");
  });

  it("binds the headline options instead of interpolating them", () => {
    const { sql, params } = buildSearchSql(config, {
      embedding: [0.1],
      text: "renewal",
      limit: 5,
      highlight: true,
    });
    expect(sql).not.toContain("StartSel");
    expect(
      params.some((param) => typeof param === "string" && param.includes("StartSel")),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------------
// Vector type and metric
// ---------------------------------------------------------------------------------

describe("vector type and metric", () => {
  it("casts the query vector and picks the halfvec opclass", () => {
    // A halfvec column can only be compared with a halfvec, and indexed by halfvec
    // ops. Getting either half wrong gives a query that still returns rows: the cast
    // makes Postgres fall back to a sequential scan, and the wrong operator class
    // makes the index unusable without saying so.
    const { sql } = buildSearchSql(makeConfig({ vectorType: "halfvec", metric: "l2" }), {
      embedding: [0.5, 0.25],
      limit: 5,
    });
    const candidates = cte(sql, "vector_candidates");

    expect(candidates).toContain("::halfvec");
    expect(candidates).not.toContain("::vector");
    // The cast is emitted twice ("$1::halfvec::halfvec") because the parameter is
    // already cast to the config's vector type before the distance expression casts it
    // again. Postgres accepts it and the plan is identical; the Python package does
    // exactly the same thing, and parity matters more here than tidiness.
    expect(count(candidates, "$1::halfvec::halfvec")).toBe(3);
  });

  it("casts to vector by default", () => {
    const { sql } = buildSearchSql(config, { embedding: [0.5], limit: 5 });
    expect(sql).toContain("$1::vector");
    expect(sql).not.toContain("halfvec");
  });

  it.each([
    ["cosine", "<=>", "vector_cosine_ops"],
    ["l2", "<->", "vector_l2_ops"],
    ["euclidean", "<->", "vector_l2_ops"],
    ["inner_product", "<#>", "vector_ip_ops"],
    ["ip", "<#>", "vector_ip_ops"],
    ["l1", "<+>", "vector_l1_ops"],
    ["manhattan", "<+>", "vector_l1_ops"],
  ] as const)("maps the metric %s to its operator", (metric, operator, ops) => {
    // Using the wrong operator ranks by the wrong distance and skips the index. The
    // failure is silent — results come back, they are just subtly worse — so the
    // mapping is pinned here rather than trusted.
    const cfg = makeConfig({ metric });
    const { sql } = buildSearchSql(cfg, { embedding: [0.5], limit: 5 });
    expect(sql).toContain(`"embedding" ${operator} $1::vector`);
    expect(opsClass(cfg)).toBe(ops);
  });

  it("excludes NULL embeddings from the candidates", () => {
    // A NULL embedding sorts as an unknown distance and would pollute the candidates.
    const { sql } = buildSearchSql(config, { embedding: [0.1], limit: 5 });
    expect(cte(sql, "vector_candidates")).toContain('"embedding" IS NOT NULL');
  });

  it("sends the vector as text for the server to cast", () => {
    // Passing pgvector's text format keeps the package driver-agnostic: no type
    // adapter has to be registered, which is what lets the library have zero runtime
    // dependencies.
    const { params } = buildSearchSql(config, { embedding: [0.1, -0.2], limit: 5 });
    expect(params[0]).toBe("[0.1,-0.2]");
  });

  it.each([
    [[1, 0, 0], "[1.0,0.0,0.0]"],
    [[0.25, -0.5, 0.75], "[0.25,-0.5,0.75]"],
    [[1e16, 1e-5], "[1e+16,1e-05]"],
    [[1e15, 0.0001], "[1000000000000000.0,0.0001]"],
    [[-0.0, 100.5], "[-0.0,100.5]"],
  ])("renders %j the way Python's repr does", (embedding, expected) => {
    // The vector literal is the one place a number reaches the statement as text, and
    // JavaScript prints a whole number as "1" where Python prints "1.0". A port that
    // gets this wrong emits SQL that is a byte different from the Python package's for
    // every query, which is exactly what the parity check exists to prevent.
    const { params } = buildSearchSql(config, { embedding, limit: 5 });
    expect(params[0]).toBe(expected);
  });
});

// ---------------------------------------------------------------------------------
// Recency
// ---------------------------------------------------------------------------------

describe("recency", () => {
  it("emits no decay when it is not configured", () => {
    const { sql } = buildSearchSql(config, { embedding: [0.1], text: "renewal", limit: 5 });
    expect(sql).not.toContain("exp(");
    expect(outputAliases(sql)).not.toContain("recency_factor");
    expect(sql).toContain("f.fused_score AS score");
  });

  it("decays the score and reports the factor", () => {
    // The factor is returned as its own column so a surprising ranking is explainable.
    const { sql, params } = buildSearchSql(
      makeConfig({ recency: { column: "published_at", halfLifeDays: 30 } }),
      { embedding: [0.1], text: "renewal", limit: 5 },
    );

    expect(sql).toContain("(f.fused_score * coalesce(exp(");
    expect(outputAliases(sql)).toContain("recency_factor");
    expect(params).toContain(30);
    // One bind slot, two references: the score and the reported factor must not be
    // able to disagree about the half-life.
    expect(count(sql, "AS recency_factor")).toBe(1);
    expect(params.filter((value) => value === 30)).toHaveLength(1);
  });

  it("leaves rows with no timestamp undecayed", () => {
    // Decaying a NULL to zero would drop every backfilled row out of the ranking,
    // which looks exactly like the search being broken.
    const { sql } = buildSearchSql(
      makeConfig({ recency: { column: "published_at", halfLifeDays: 30 } }),
      { embedding: [0.1], limit: 5 },
    );
    const match = /(coalesce\(exp\([\s\S]+?\), 1\.0\)) AS recency_factor/.exec(sql);
    expect(match, sql).not.toBeNull();
    const decay = match![1]!;

    expect(decay.startsWith("coalesce(")).toBe(true);
    expect(decay.endsWith(", 1.0)")).toBe(true);
    // A future timestamp must not amplify the score past 1.0 either.
    expect(decay).toContain('greatest(extract(epoch from (now() - "published_at")), 0)');
    // Half-life is expressed in days, so the epoch seconds are scaled by a day.
    expect(decay).toContain("* 86400.0");
  });

  it("quotes the recency column", () => {
    const { sql } = buildSearchSql(
      makeConfig({ recency: { column: "published_at", halfLifeDays: 7 } }),
      { embedding: [0.1], limit: 5 },
    );
    expect(sql).toContain('now() - "published_at"');
  });
});

// ---------------------------------------------------------------------------------
// Limits, offsets and the candidate budget
// ---------------------------------------------------------------------------------

describe("limits and the candidate budget", () => {
  it("extends the final limit by nearMiss", () => {
    // The rows that just missed the cut are usually why a search "failed".
    const { sql, params } = buildSearchSql(config, {
      embedding: [0.1],
      limit: 10,
      nearMiss: 5,
      offset: 20,
    });
    const match = /LIMIT \$(\d+) OFFSET \$(\d+)/.exec(sql);
    expect(match, sql).not.toBeNull();
    expect(params[Number(match![1]) - 1]).toBe(15);
    expect(params[Number(match![2]) - 1]).toBe(20);
  });

  it("raises the candidate limit to cover limit plus nearMiss", () => {
    // Fusing fewer candidates than we return truncates the result before ranking.
    const { sql, params } = buildSearchSql(makeConfig({ candidateLimit: 5 }), {
      embedding: [0.1],
      text: "renewal",
      limit: 10,
      nearMiss: 3,
    });
    expect(boundLimit(cte(sql, "vector_candidates"), params)).toBe(13);
    expect(boundLimit(cte(sql, "text_candidates"), params)).toBe(13);
  });

  it("leaves a large enough candidate limit alone", () => {
    const { sql, params } = buildSearchSql(makeConfig({ candidateLimit: 200 }), {
      embedding: [0.1],
      limit: 10,
      nearMiss: 3,
    });
    expect(boundLimit(cte(sql, "vector_candidates"), params)).toBe(200);
  });

  it("lets the candidateLimit argument override the config", () => {
    const { sql, params } = buildSearchSql(makeConfig({ candidateLimit: 50 }), {
      embedding: [0.1],
      limit: 10,
      candidateLimit: 120,
    });
    expect(boundLimit(cte(sql, "vector_candidates"), params)).toBe(120);
  });

  it("falls back to the config for a candidateLimit of zero", () => {
    // Documenting current behaviour, not endorsing it: zero is treated as "not
    // supplied" rather than as an out-of-range value, the way the Python package does.
    const { sql, params } = buildSearchSql(makeConfig({ candidateLimit: 50 }), {
      embedding: [0.1],
      limit: 10,
      candidateLimit: 0,
    });
    expect(boundLimit(cte(sql, "vector_candidates"), params)).toBe(50);
  });

  it("orders and limits each candidate CTE independently", () => {
    // Both signals must contribute a full candidate list, ranked on their own terms.
    const { sql } = buildSearchSql(config, { embedding: [0.1], text: "renewal", limit: 5 });
    const vector = cte(sql, "vector_candidates");
    const text = cte(sql, "text_candidates");
    expect(vector).toContain('ORDER BY "embedding" <=>');
    expect(vector).toContain("rank() OVER (ORDER BY");
    expect(text).toContain("DESC");
    expect(text).toContain("rank() OVER (ORDER BY");
  });

  it("orders results by the final score with a stable tiebreak", () => {
    // Without the id tiebreak, paging through equal scores can repeat or skip rows.
    const { sql } = buildSearchSql(config, { embedding: [0.1], text: "renewal", limit: 5 });
    expect(sql).toContain("ORDER BY score DESC, f.id");
  });
});

// ---------------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------------

describe("argument validation", () => {
  it("requires at least one signal", () => {
    expect(() => buildSearchSql(config, { limit: 5 })).toThrow(/embedding or text/);
  });

  it.each([0, -1, -100])("rejects the limit %i", (limit) => {
    expect(() => buildSearchSql(config, { embedding: [0.1], limit })).toThrow(
      /limit must be >= 1/,
    );
  });

  it.each([-1, -50])("rejects the offset %i", (offset) => {
    expect(() => buildSearchSql(config, { embedding: [0.1], limit: 5, offset })).toThrow(
      /offset must be >= 0/,
    );
  });

  it.each([["a"], [null], [{}]])("rejects the non-numeric embedding %j", (value) => {
    // Catching this here beats a driver-specific cast error from the server.
    expect(() =>
      buildSearchSql(config, { embedding: [value] as never, limit: 5 }),
    ).toThrow(/sequence of numbers/);
  });

  it("builds an empty embedding that pgvector will reject", () => {
    // Documenting current behaviour: the builder has no idea how many dimensions the
    // column has, so it does not pretend to validate the length.
    const { params } = buildSearchSql(config, { embedding: [], limit: 5 });
    expect(params[0]).toBe("[]");
  });

  it("interpolates the language without validating it", () => {
    // Documenting a real gap, not endorsing it: language, queryParser and rankFunction
    // are typed but never checked at runtime, and all three are interpolated straight
    // into the statement. A config built from user input is therefore an injection
    // surface, unlike every other field. The Python package has the same gap and the
    // two have to agree until it is closed in both.
    const { sql } = buildSearchSql(
      makeConfig({ tsvectorColumn: null, language: "english', 'injected" }),
      { text: "renewal", limit: 5 },
    );
    expect(sql).toContain("'english', 'injected'");
  });
});

// ---------------------------------------------------------------------------------
// Golden snapshot
// ---------------------------------------------------------------------------------

/**
 * One fully-featured query, spelled out rather than built from a fixture.
 *
 * A snapshot that depends on a shared fixture changes meaning whenever somebody tunes
 * the fixture for an unrelated test, so this one owns its inputs. It is the same query
 * the Python suite pins, argument for argument.
 */
function canonicalQuery() {
  const cfg: Config = {
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
  return buildSearchSql(cfg, {
    embedding: [0.25, -0.5, 0.75],
    text: 'renewal "notice period" -pricing',
    limit: 10,
    offset: 20,
    filters: { tenant_id: 42, lang: ["en", "de"] },
    nearMiss: 3,
    highlight: true,
  });
}

describe("the golden snapshot", () => {
  it("reproduces the Python package's statement byte for byte", () => {
    // This snapshot is the contract between the two ports. Two ports that agree on the
    // public API but disagree on the SQL they emit are two different libraries with one
    // name, and the difference would only ever surface as "the JS results are a bit
    // worse", which nobody can debug.
    //
    // A diff here is either a deliberate change to the query — made in both packages,
    // with the Python golden file regenerated — or a bug that just escaped every other
    // test in this file.
    const { sql } = canonicalQuery();
    expect(sql + "\n").toBe(readFileSync(GOLDEN, "utf8"));
  });

  it("binds the same values in the same order", () => {
    // The bind values are half the contract; SQL alone would not catch a reordering.
    const { params } = canonicalQuery();
    expect(params).toEqual([
      "[0.25,-0.5,0.75]",
      42,
      ["en", "de"],
      80,
      "renewal",
      "notice period",
      "pricing",
      42,
      ["en", "de"],
      80,
      1.5,
      60,
      1.0,
      30.0,
      "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=8, MaxWords=30",
      13,
      20,
    ]);
  });

  it("is deterministic", () => {
    // A snapshot test is worthless if the builder emits set-ordered output.
    const first = canonicalQuery();
    const second = canonicalQuery();
    expect(first.sql).toBe(second.sql);
    expect(first.params).toEqual(second.params);
  });
});
