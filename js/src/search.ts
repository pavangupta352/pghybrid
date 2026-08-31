/**
 * Running the generated statement, and shaping what comes back.
 *
 * pghybrid never opens a connection. A {@link HybridSearch} is built around an
 * `execute` callable that takes `(sql, params)` and returns rows, which is what lets
 * the package work with node-postgres, postgres.js, Drizzle or the Supabase client
 * without importing any of them and without holding an opinion about pooling,
 * transactions or retries:
 *
 * ```ts
 * const search = new HybridSearch(config, (sql, params) =>
 *   pool.query(sql, params).then((result) => result.rows));
 * ```
 *
 * The row shaping is the part worth being careful about. A row found by only one
 * signal has a NULL rank for the other, and the natural implementation —
 * `Number(row.text_rank)` — turns that into NaN on exactly the rows hybrid search
 * exists to surface. Every conversion here tolerates NULL and says so.
 */

import type { Config, FusionMethod } from "./config.js";
import type { BuiltQuery, Filters } from "./sql.js";
import { buildSearchSql } from "./sql.js";

/** One row as the driver returned it. Anything object-like survives {@link rowMapping}. */
export type Row = Record<string, unknown>;

/**
 * What the caller's `execute` is expected to be. It may be synchronous, which keeps
 * the door open for drivers that are.
 */
export type Executor = (
  sql: string,
  params: unknown[],
) => Promise<Iterable<unknown>> | Iterable<unknown>;

/**
 * Columns the fused query always produces. Everything else in a row came from the
 * user's table and is handed back untouched in {@link SearchResult.row}.
 */
const SIGNAL_COLUMNS = new Set([
  "id",
  "score",
  "fused_score",
  "vector_rank",
  "vector_distance",
  "vector_contribution",
  "text_rank",
  "text_score",
  "text_contribution",
  "recency_factor",
  "highlight",
]);

/** Which signals retrieved a row. */
export type MatchedBy = "both" | "vector" | "text" | "none";

/**
 * One ranked row, with the arithmetic that produced its position kept intact.
 *
 * The decomposition is not decoration. When a search result looks wrong the only
 * useful question is which signal put it there, and a bare `{ id, score }` cannot
 * answer it. Both ranks, both raw scores and both fused contributions travel with
 * every row so the answer is always one property away.
 *
 * `vectorRank` and `textRank` are null when that signal did not retrieve the row at
 * all, which is different from retrieving it last.
 */
export interface SearchResult {
  id: unknown;
  score: number;
  fusedScore: number;
  vectorRank: number | null;
  vectorDistance: number | null;
  vectorContribution: number;
  textRank: number | null;
  textScore: number | null;
  textContribution: number;
  recencyFactor: number | null;
  highlight: string | null;
  /** Which signals retrieved this row: `both`, `vector`, `text` or `none`. */
  matchedBy: MatchedBy;
  /** The columns copied through from the table (`textColumn` plus `extraColumns`). */
  row: Row;
}

/**
 * Coerce one driver row into a plain object.
 *
 * Drivers disagree about what a row is: node-postgres returns plain objects,
 * postgres.js returns array-like rows with named properties, and some query builders
 * return a Map. Accepting all of them here is what keeps the `execute` callable a
 * one-liner instead of an adapter the user has to write.
 */
export function rowMapping(row: unknown): Row {
  if (row instanceof Map) {
    return Object.fromEntries(row.entries()) as Row;
  }
  if (typeof row === "object" && row !== null && !Array.isArray(row)) {
    return row as Row;
  }
  throw new TypeError(
    `execute() returned ${row === null ? "null" : typeof row} rows, which are not ` +
      "object-like. Return one object per row with the column names as keys, which " +
      "node-postgres, postgres.js and Drizzle all do by default.",
  );
}

/**
 * Number conversion that passes NULL through instead of turning it into NaN.
 *
 * Also normalises the strings a driver returns for bigint and numeric columns — the
 * rank() window function is a bigint, so node-postgres hands back "2" rather than 2 —
 * so a caller never has to think about which driver produced a score.
 */
export function asFloat(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "number") {
    return value;
  }
  const converted = Number(value);
  if (Number.isNaN(converted)) {
    throw new TypeError(
      `expected a number from the database, got ${JSON.stringify(value) ?? String(value)}`,
    );
  }
  return converted;
}

function asInt(value: unknown): number | null {
  const converted = asFloat(value);
  return converted === null ? null : Math.trunc(converted);
}

/** A contribution that must be a number: absent signals contribute zero, not NULL. */
function asScore(value: unknown): number {
  return asFloat(value) ?? 0;
}

/**
 * Build a {@link SearchResult} from one driver row.
 *
 * Anything the query added beyond the ranking columns is passed through in `row`
 * rather than dropped, because the caller usually needs the title next to the score.
 */
export function resultFromRow(row: unknown): SearchResult {
  const mapping = rowMapping(row);
  if (!("id" in mapping)) {
    throw new Error(
      "no 'id' column in the result row. This happens when execute() runs a statement " +
        "pghybrid did not build; pass the sql and params from HybridSearch.buildQuery() " +
        "through unchanged.",
    );
  }

  const vectorRank = asInt(mapping["vector_rank"]);
  const textRank = asInt(mapping["text_rank"]);
  const passthrough: Row = {};
  for (const [key, value] of Object.entries(mapping)) {
    if (!SIGNAL_COLUMNS.has(key)) {
      passthrough[key] = value;
    }
  }

  return {
    id: mapping["id"],
    score: asScore(mapping["score"]),
    fusedScore: asScore(mapping["fused_score"]),
    vectorRank,
    vectorDistance: asFloat(mapping["vector_distance"]),
    vectorContribution: asScore(mapping["vector_contribution"]),
    textRank,
    textScore: asFloat(mapping["text_score"]),
    textContribution: asScore(mapping["text_contribution"]),
    recencyFactor: asFloat(mapping["recency_factor"]),
    highlight: (mapping["highlight"] as string | null | undefined) ?? null,
    matchedBy: matchedBy(vectorRank, textRank),
    row: passthrough,
  };
}

function matchedBy(vectorRank: number | null, textRank: number | null): MatchedBy {
  if (vectorRank !== null && textRank !== null) {
    return "both";
  }
  if (vectorRank !== null) {
    return "vector";
  }
  if (textRank !== null) {
    return "text";
  }
  return "none";
}

/** Shape a whole result set. The ordering the database produced is preserved. */
export function resultsFromRows(rows: Iterable<unknown> | null | undefined): SearchResult[] {
  if (rows === null || rows === undefined) {
    return [];
  }
  return Array.from(rows, resultFromRow);
}

/** Everything a caller may vary from one search to the next. */
export interface SearchOptions {
  embedding?: readonly number[] | null;
  limit?: number;
  offset?: number;
  filters?: Filters | null;
  candidateLimit?: number | null;
  /** Extra rows past `limit`, for showing the ones that just missed the cut. */
  nearMiss?: number;
  highlight?: boolean;
  fusion?: FusionMethod | null;
}

/**
 * Treat a blank search box as no text signal at all.
 *
 * An empty tsquery matches nothing, so passing `""` through would build a text CTE
 * that can only ever be empty and a ts_headline call over it. Dropping the signal
 * gives the same rows for less work, and makes a blank query with no embedding fail
 * loudly rather than return an empty list that looks like a relevance problem.
 */
function normaliseText(text: string | null | undefined): string | null {
  if (text === null || text === undefined) {
    return null;
  }
  return text.trim() ? text : null;
}

/**
 * Hybrid search over one table, driven by an `execute` callable.
 *
 * `execute(sql, params)` must run the statement and return the rows as objects.
 * Everything else — connecting, pooling, retrying, tracing — stays in the caller's
 * code where it belongs.
 */
export class HybridSearch {
  readonly config: Config;
  readonly execute: Executor;

  constructor(config: Config, execute: Executor) {
    if (typeof config !== "object" || config === null) {
      throw new TypeError(`config must be a pghybrid Config object, got ${typeof config}`);
    }
    if (typeof execute !== "function") {
      throw new TypeError(
        "execute must be callable as execute(sql, params). For node-postgres: " +
          "(sql, params) => pool.query(sql, params).then((r) => r.rows)",
      );
    }
    this.config = config;
    this.execute = execute;
  }

  /**
   * The statement {@link HybridSearch.search} would run, without running it.
   *
   * Worth exposing: the fastest way to debug a ranking is to paste the query into psql
   * and edit it, and the fastest way to trust a library is to read what it sends.
   */
  buildQuery(text?: string | null, options: SearchOptions = {}): BuiltQuery {
    return buildSearchSql(this.config, {
      embedding: options.embedding ?? null,
      text: normaliseText(text),
      limit: options.limit ?? 10,
      offset: options.offset ?? 0,
      filters: options.filters ?? null,
      candidateLimit: options.candidateLimit ?? null,
      nearMiss: options.nearMiss ?? 0,
      highlight: options.highlight ?? false,
      fusion: options.fusion ?? null,
    });
  }

  /**
   * Rank rows by both signals at once.
   *
   * Passing only `text` runs a pure full-text search and only `embedding` a pure
   * vector search, both returning the same shape, which is what makes comparing the
   * three an apples-to-apples exercise.
   */
  async search(text?: string | null, options: SearchOptions = {}): Promise<SearchResult[]> {
    const { sql, params } = this.buildQuery(text, options);
    return resultsFromRows(await this.execute(sql, params));
  }
}
