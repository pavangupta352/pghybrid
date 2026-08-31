/**
 * Unit tests for the runtime wrapper.
 *
 * No database here either: the executor is a function, so everything the client does
 * around the driver — building the statement, shaping the rows, tolerating the NULLs
 * that a single-signal match leaves behind — is testable without one.
 */

import { describe, expect, it } from "vitest";

import type { Config } from "../src/config.js";
import { HybridSearch, resultFromRow, resultsFromRows, rowMapping } from "../src/search.js";

const CONFIG: Config = {
  table: "chunks",
  textColumn: "content",
  vectorColumn: "embedding",
  tsvectorColumn: "fts",
  extraColumns: ["title"],
  filterColumns: ["tenant_id"],
};

/** A row shaped the way the generated query returns one. */
function hybridRow(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 7,
    score: 0.032258,
    fused_score: 0.032258,
    vector_rank: "2",
    vector_distance: 0.039,
    vector_contribution: 0.016129,
    text_rank: "2",
    text_score: 0.3,
    text_contribution: 0.016129,
    title: "Termination for convenience",
    content: "sixty days written notice",
    ...overrides,
  };
}

describe("HybridSearch", () => {
  it("passes the built statement to the executor unchanged", async () => {
    const seen: { sql: string; params: unknown[] }[] = [];
    const search = new HybridSearch(CONFIG, (sql, params) => {
      seen.push({ sql, params });
      return [hybridRow()];
    });

    await search.search("renewal notice period", { embedding: [0.1, 0.2], limit: 3 });

    expect(seen).toHaveLength(1);
    const built = search.buildQuery("renewal notice period", {
      embedding: [0.1, 0.2],
      limit: 3,
    });
    expect(seen[0]!.sql).toBe(built.sql);
    expect(seen[0]!.params).toEqual(built.params);
  });

  it("defaults to ten results", () => {
    const { sql, params } = new HybridSearch(CONFIG, () => []).buildQuery("renewal", {
      embedding: [0.1],
    });
    const match = /LIMIT \$(\d+) OFFSET \$(\d+)/.exec(sql);
    expect(params[Number(match![1]) - 1]).toBe(10);
    expect(params[Number(match![2]) - 1]).toBe(0);
  });

  it.each(["", "   ", "\n\t"])("drops a blank search box (%j) as a signal", (text) => {
    // An empty tsquery matches nothing, so passing "" through would build a text CTE
    // that can only ever be empty. Dropping the signal gives the same rows for less
    // work, and makes a blank query with no embedding fail loudly rather than return
    // an empty list that looks like a relevance problem.
    const search = new HybridSearch(CONFIG, () => []);
    const { sql } = search.buildQuery(text, { embedding: [0.1] });
    expect(sql).not.toContain("text_candidates");

    expect(() => search.buildQuery(text)).toThrow(/embedding or text/);
  });

  it("keeps a search box that only looks blank", () => {
    const { sql } = new HybridSearch(CONFIG, () => []).buildQuery("  renewal  ", {
      embedding: [0.1],
    });
    expect(sql).toContain("text_candidates");
  });

  it("accepts a synchronous executor", async () => {
    const results = await new HybridSearch(CONFIG, () => [hybridRow()]).search("renewal", {
      embedding: [0.1],
    });
    expect(results).toHaveLength(1);
    expect(results[0]!.id).toBe(7);
  });

  it("rejects an executor that is not callable", () => {
    expect(() => new HybridSearch(CONFIG, null as never)).toThrow(/execute\(sql, params\)/);
  });
});

describe("row shaping", () => {
  it("keeps the arithmetic that produced the ranking", () => {
    // When a result looks wrong the only useful question is which signal put it there,
    // and a bare { id, score } cannot answer it.
    const result = resultFromRow(hybridRow());
    expect(result.score).toBeCloseTo(0.032258);
    expect(result.vectorRank).toBe(2);
    expect(result.textRank).toBe(2);
    expect(result.vectorContribution).toBeCloseTo(0.016129);
    expect(result.textContribution).toBeCloseTo(0.016129);
    expect(result.matchedBy).toBe("both");
  });

  it("converts the strings a driver returns for bigint columns", () => {
    // rank() is a bigint, and node-postgres hands bigints back as strings rather than
    // risk a silent precision loss. A rank of "2" that stays a string compares wrong
    // against every number the caller puts next to it.
    const result = resultFromRow(hybridRow({ vector_rank: "12", text_score: "0.5" }));
    expect(result.vectorRank).toBe(12);
    expect(result.textScore).toBe(0.5);
  });

  it("passes a missing signal through as null rather than NaN", () => {
    // This is the row hybrid search exists to surface: found by one signal, invisible
    // to the other. Number(null) is 0 and Number(undefined) is NaN, and both would be
    // read as a real rank.
    const result = resultFromRow(
      hybridRow({ text_rank: null, text_score: null, text_contribution: 0 }),
    );
    expect(result.textRank).toBeNull();
    expect(result.textScore).toBeNull();
    expect(result.textContribution).toBe(0);
    expect(result.matchedBy).toBe("vector");
  });

  it.each([
    [{ vector_rank: null, text_rank: null }, "none"],
    [{ vector_rank: 1, text_rank: null }, "vector"],
    [{ vector_rank: null, text_rank: 1 }, "text"],
    [{ vector_rank: 1, text_rank: 1 }, "both"],
  ])("reports %j as matched by %s", (ranks, expected) => {
    expect(resultFromRow(hybridRow(ranks)).matchedBy).toBe(expected);
  });

  it("hands the table's own columns back untouched", () => {
    // The caller usually needs the title next to the score, so passthrough columns are
    // kept rather than dropped — and kept out of the ranking fields so the two can
    // never collide.
    const result = resultFromRow(hybridRow());
    expect(result.row).toEqual({
      title: "Termination for convenience",
      content: "sixty days written notice",
    });
    expect(result.row).not.toHaveProperty("score");
  });

  it("explains a row that did not come from this library", () => {
    expect(() => resultFromRow({ title: "no id here" })).toThrow(/buildQuery/);
  });

  it("refuses rows that are not object-like", () => {
    expect(() => rowMapping([1, 2, 3])).toThrow(/object-like/);
    expect(() => rowMapping(null)).toThrow(/object-like/);
  });

  it("reads a Map row", () => {
    expect(rowMapping(new Map([["id", 3]]))).toEqual({ id: 3 });
  });

  it("preserves the order the database produced", () => {
    const rows = [hybridRow({ id: 1 }), hybridRow({ id: 2 }), hybridRow({ id: 3 })];
    expect(resultsFromRows(rows).map((result) => result.id)).toEqual([1, 2, 3]);
  });

  it("treats no rows as no results", () => {
    expect(resultsFromRows(null)).toEqual([]);
    expect(resultsFromRows([])).toEqual([]);
  });
});
