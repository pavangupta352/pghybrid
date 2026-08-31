/**
 * Unit tests for query parsing and the tsquery it produces.
 *
 * Two layers are covered here. The first is `parseQuery`, which splits a search box
 * into terms. The second is what the SQL builder does with those terms, because the
 * tokenising only matters insofar as it produces a tsquery that ranks the way a person
 * expects — and because the most damaging bug in this area is invisible at the parsed
 * level and only appears in the generated expression.
 */

import { describe, expect, it } from "vitest";

import type { Config } from "../src/config.js";
import { buildSearchSql } from "../src/sql.js";
import { parseQuery } from "../src/textquery.js";

function makeConfig(overrides: Partial<Config> = {}): Config {
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

/** The expression the text_query CTE assigns to tsq. */
function tsqueryExpression(sql: string): string {
  const match = /text_query AS \(\n {4}SELECT (.*) AS tsq\n\)/.exec(sql);
  expect(match, `no text_query CTE in:\n${sql}`).not.toBeNull();
  return match![1]!;
}

function terms(text: string): { positive: string[]; negative: string[] } {
  const parsed = parseQuery(text);
  return { positive: parsed.positive, negative: parsed.negative };
}

describe("parseQuery", () => {
  it("splits plain terms on whitespace", () => {
    expect(terms("renewal notice period")).toEqual({
      positive: ["renewal", "notice", "period"],
      negative: [],
    });
  });

  it("keeps quoted phrases whole", () => {
    // A phrase is one term, so it can be handed to the parser as a phrase.
    expect(terms('renewal "notice period" clause')).toEqual({
      positive: ["renewal", "notice period", "clause"],
      negative: [],
    });
  });

  it("excludes a term after a leading dash", () => {
    expect(terms("renewal -pricing")).toEqual({ positive: ["renewal"], negative: ["pricing"] });
  });

  it("excludes a whole quoted phrase", () => {
    expect(terms('renewal -"legacy plan"')).toEqual({
      positive: ["renewal"],
      negative: ["legacy plan"],
    });
  });

  it("parses the documented example as documented", () => {
    // The doc comment on parseQuery is the API contract most people will read.
    expect(terms('renewal "notice period" -pricing')).toEqual({
      positive: ["renewal", "notice period"],
      negative: ["pricing"],
    });
  });

  it("does not treat a dash inside a word as a negation", () => {
    // Hyphenated words are common; only a leading dash excludes.
    expect(terms("multi-tenant end-of-life")).toEqual({
      positive: ["multi-tenant", "end-of-life"],
      negative: [],
    });
  });

  it.each(["or", "and", "OR", "And", "aNd"])("drops the bare boolean word %s", (noise) => {
    // Under ANY semantics the OR is already implied. Searching for the literal word
    // "or" would also pollute the ranking, since it is a stopword in most
    // configurations and contributes nothing but a wasted parser call.
    expect(terms(`renewal ${noise} termination`)).toEqual({
      positive: ["renewal", "termination"],
      negative: [],
    });
  });

  it("still negates a boolean word that was explicitly excluded", () => {
    // "-or" is an explicit instruction, not the noise word the tokeniser drops. The
    // noise filter has to run only on bare terms, or a user excluding a literal word
    // silently gets no exclusion at all.
    expect(terms("renewal -or")).toEqual({ positive: ["renewal"], negative: ["or"] });
    expect(terms("-and")).toEqual({ positive: [], negative: ["and"] });
  });

  it("drops a quoted boolean word too", () => {
    // Documenting current behaviour, and matching the Python package exactly.
    expect(terms('"or"')).toEqual({ positive: [], negative: [] });
    expect(terms('"and or"')).toEqual({ positive: ["and or"], negative: [] });
  });

  it.each(["", "   ", "\t\n ", null, undefined])(
    "yields nothing for empty input (%j)",
    (text) => {
      // null is accepted because callers pass whatever the search box gave them.
      const parsed = parseQuery(text);
      expect(parsed.positive).toEqual([]);
      expect(parsed.negative).toEqual([]);
      expect(parsed.isEmpty).toBe(true);
    },
  );

  it("contributes no term for empty quotes", () => {
    expect(terms('renewal ""')).toEqual({ positive: ["renewal"], negative: [] });
    expect(terms('-"" renewal')).toEqual({ positive: ["renewal"], negative: [] });
  });

  it("keeps punctuation-only input as a term", () => {
    // Documenting current behaviour: the tokeniser does not judge term content.
    // Postgres' parser turns punctuation into an empty tsquery, so the term matches
    // nothing and costs one parser call. Filtering it here would mean deciding what
    // counts as a word in every language the library supports.
    expect(terms("!!! ???")).toEqual({ positive: ["!!!", "???"], negative: [] });
  });

  it("reports a query of only exclusions as non-empty", () => {
    const parsed = parseQuery("-pricing -legacy");
    expect(parsed.positive).toEqual([]);
    expect(parsed.negative).toEqual(["pricing", "legacy"]);
    expect(parsed.isEmpty).toBe(false);
  });

  it("passes unicode terms through intact", () => {
    // Accents, CJK and emoji all survive. Any normalisation belongs to the text search
    // configuration, not to a tokeniser that has no idea which language it is looking
    // at.
    expect(terms('café 日本語 "kündigung frist" -naïve 🙂')).toEqual({
      positive: ["café", "日本語", "kündigung frist", "🙂"],
      negative: ["naïve"],
    });
  });

  it("tokenises the same string the same way twice", () => {
    // A module-level /g regex keeps its lastIndex between calls, which would make
    // every second parse start from the middle of the string.
    expect(terms("renewal notice")).toEqual(terms("renewal notice"));
  });
});

describe("the tsquery the builder generates", () => {
  it("ORs one parser call per term in any mode", () => {
    // With Postgres' native AND, a four-word query usually matches nothing, the text
    // candidate list comes back empty, and the fusion degrades to vector-only search
    // without reporting that anything went wrong. Precision comes back through
    // ranking: ts_rank_cd already scores a document matching three terms above one
    // matching one.
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      text: "renewal notice period",
      limit: 5,
    });
    const tsq = tsqueryExpression(sql);

    expect(tsq.split("websearch_to_tsquery('english', $").length - 1).toBe(3);
    expect(tsq.split(" || ").length - 1).toBe(2);
    expect(tsq).not.toContain(" && ");
    expect(params.slice(0, 3)).toEqual(["renewal", "notice", "period"]);
  });

  it("binds each term separately", () => {
    // One term per placeholder, so no term can be smuggled in as query syntax.
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      text: "renewal termination",
      limit: 5,
    });
    expect(tsqueryExpression(sql)).toBe(
      "(websearch_to_tsquery('english', $1) || websearch_to_tsquery('english', $2))",
    );
    expect(params.slice(0, 2)).toEqual(["renewal", "termination"]);
  });

  it("needs no parentheses for a single term", () => {
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      text: "renewal",
      limit: 5,
    });
    expect(tsqueryExpression(sql)).toBe("websearch_to_tsquery('english', $1)");
    expect(params[0]).toBe("renewal");
  });

  it("makes exactly one parser call with the whole string in all mode", () => {
    // AND semantics are the right default for a filter, and are still one call away.
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "all" }), {
      text: "renewal notice -pricing",
      limit: 5,
    });
    expect(tsqueryExpression(sql)).toBe("websearch_to_tsquery('english', $1)");
    expect(params[0]).toBe("renewal notice -pricing");
  });

  it("constrains both signals with an exclusion, not just the keyword one", () => {
    // A leading dash is a statement about the answer, so both halves have to honour it.
    // The tempting implementation puts the exclusion in the tsquery and stops, because
    // that is where the parser already understands it. The vector half then never hears
    // about it and happily returns the excluded rows: they drop out of the text
    // candidates, so they arrive with a vector rank and no text rank, and RRF pays the
    // best vector hit 1/(k+1) — the largest single contribution it can award.
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      embedding: [0.1, 0.2],
      text: "renewal notice -pricing",
      limit: 5,
    });
    const vectorCte = sql.slice(sql.indexOf("vector_candidates AS ("), sql.indexOf("text_query AS ("));
    const textCte = sql.slice(sql.indexOf("text_candidates AS ("), sql.indexOf("scored AS ("));

    const exclusion = `NOT coalesce("content_tsv" @@ websearch_to_tsquery('english', $2), false)`;
    expect(vectorCte, "the vector half can still return the excluded rows").toContain(exclusion);
    expect(textCte).toContain(exclusion);
    // Inside the candidate subquery and before its LIMIT, not wrapped around the fusion
    // afterwards: excluding rows after the cut-off returns fewer than the caller asked for.
    expect(vectorCte.indexOf(exclusion)).toBeLessThan(vectorCte.indexOf("LIMIT"));

    // The tsquery itself carries only what the user asked to find.
    expect(tsqueryExpression(sql)).not.toContain("!!");
    // Negation is an operator in the statement, never a character left inside a bound
    // value: the naive form would hand the raw string to a single parser call.
    expect(params).toContain("pricing");
    expect(params).not.toContain("renewal notice -pricing");
    expect(params).not.toContain("-pricing");
  });

  it("applies every exclusion", () => {
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      embedding: [0.1, 0.2],
      text: "renewal -pricing -legacy",
      limit: 5,
    });
    const vectorCte = sql.slice(sql.indexOf("vector_candidates AS ("), sql.indexOf("text_query AS ("));
    expect(vectorCte).toContain("$2) || websearch_to_tsquery('english', $3)");
    expect(params.slice(1, 4)).toEqual(["pricing", "legacy", 50]);
  });

  it("renders the exclusion once and references it twice", () => {
    // The predicate appears in two CTEs, so this is the one place in the statement where
    // a single logical value is genuinely referenced twice. Both paramstyles have to be
    // right: numbered styles reuse the placeholder, positional styles repeat the value.
    const numeric = buildSearchSql(makeConfig({ textMatch: "any" }), {
      embedding: [0.1, 0.2],
      text: "renewal -pricing",
      limit: 5,
    });
    expect(numeric.sql.split("$2").length - 1).toBe(2);
    expect(numeric.params.filter((v) => v === "pricing")).toHaveLength(1);

    const pyformat = buildSearchSql(makeConfig({ textMatch: "any", paramStyle: "pyformat" }), {
      embedding: [0.1, 0.2],
      text: "renewal -pricing",
      limit: 5,
    });
    expect(pyformat.params.filter((v) => v === "pricing")).toHaveLength(2);
  });

  it("does not exclude a null tsvector by a term it cannot contain", () => {
    // NULL @@ q is NULL and NOT NULL is NULL, which would drop the row. A row with no
    // tsvector contains no words, so it contains no excluded word either.
    const { sql } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      embedding: [0.1, 0.2],
      text: "renewal -pricing",
      limit: 5,
    });
    expect(sql).toContain("NOT coalesce(");
    expect(sql).toContain(", false)");
  });

  it("has no keyword signal for a query of only exclusions", () => {
    // Handing "-pricing" to the parser yields !'pricing', which matches almost the whole
    // table. ts_rank_cd scores a pure negation identically for every row, so the keyword
    // half would contribute an arbitrary order — at full weight — that reshuffles the
    // vector results for no reason. The exclusion still applies to what remains.
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      embedding: [0.1, 0.2],
      text: "-pricing",
      limit: 5,
    });
    expect(sql).not.toContain("text_candidates");
    expect(sql).toContain("NOT coalesce(");
    expect(params[1]).toBe("pricing");
    expect(params).not.toContain("-pricing");
  });

  it("says what is missing when there is nothing to rank", () => {
    expect(() =>
      buildSearchSql(makeConfig({ textMatch: "any" }), { text: "-pricing", limit: 5 }),
    ).toThrow(/only excludes terms/);

    // Same choice normaliseText makes for a blank search box: say what is wrong, because
    // a caller handed [] goes looking in the ranking rather than at the query.
    expect(() =>
      buildSearchSql(makeConfig({ textMatch: "any" }), { text: "and or", limit: 5 }),
    ).toThrow(/no searchable terms/);

    // With an embedding there is still a search to run; it is simply vector-only.
    const { sql } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      embedding: [0.1, 0.2],
      text: "and or",
      limit: 5,
    });
    expect(sql).not.toContain("text_candidates");
  });

  it("uses the configured parser and language", () => {
    const { sql } = buildSearchSql(
      makeConfig({ textMatch: "any", queryParser: "plainto_tsquery", language: "german" }),
      { text: "kündigung frist", limit: 5 },
    );
    const tsq = tsqueryExpression(sql);
    expect(tsq.split("plainto_tsquery('german', $").length - 1).toBe(2);
    expect(tsq).not.toContain("websearch_to_tsquery");
  });

  it("keeps a quoted phrase as one parser call, adjacency and all its faults", () => {
    // Documenting a real gap, not endorsing it, and matching Python exactly: the
    // phrase survives as a single term and a single bound value, but the quotes are
    // stripped, so websearch_to_tsquery('english', 'notice period') is 'notic' &
    // 'period' rather than the adjacency the user asked for by quoting.
    const { sql, params } = buildSearchSql(makeConfig({ textMatch: "any" }), {
      text: 'renewal "notice period"',
      limit: 5,
    });
    expect(tsqueryExpression(sql).split("websearch_to_tsquery('english', $").length - 1).toBe(2);
    expect(params.slice(0, 2)).toEqual(["renewal", "notice period"]);
  });
});

describe("repeats and long queries", () => {
  // Two failure modes that only appear when someone pastes text into a search box.

  it("collapses repeated terms", () => {
    // `a | a` is `a`, so a repeat only makes the statement bigger.
    expect(parseQuery("renewal Renewal RENEWAL notice renewal").positive).toEqual([
      "renewal",
      "notice",
    ]);
  });

  it("keeps the first spelling of a repeated term", () => {
    // Deduplication folds case to compare but must not rewrite what it keeps.
    expect(parseQuery("Renewal renewal").positive).toEqual(["Renewal"]);
  });

  it("deduplicates positive and negative terms separately", () => {
    const parsed = parseQuery("renewal renewal -pricing -pricing");
    expect(parsed.positive).toEqual(["renewal"]);
    expect(parsed.negative).toEqual(["pricing"]);
  });

  it("caps a pasted document before it blows the parser stack", () => {
    // Past roughly 4,200 OR-ed parser calls Postgres reports a stack depth limit, which
    // reads like an internal error rather than "that query was too long".
    const query = Array.from({ length: 10_000 }, (_, i) => `term${i}`).join(" ");
    const { sql, params } = buildSearchSql(makeConfig({ tsvectorColumn: "fts" }), {
      text: query,
      limit: 5,
    });
    expect(sql.split("websearch_to_tsquery").length - 1).toBe(200);
    expect(params.filter((p) => typeof p === "string" && p.startsWith("term")).length).toBe(200);
  });

  it("takes the cap from the config and validates it", () => {
    const { sql } = buildSearchSql(makeConfig({ maxQueryTerms: 3 }), {
      text: "a b c d e f",
      limit: 5,
    });
    expect(sql.split("websearch_to_tsquery").length - 1).toBe(3);
    expect(() => buildSearchSql(makeConfig({ maxQueryTerms: 0 }), { text: "a", limit: 5 })).toThrow(
      /maxQueryTerms/,
    );
  });
});
