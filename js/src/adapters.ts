/**
 * One-line wiring for the drivers people actually have.
 *
 * `HybridSearch` takes an `execute` callable, which keeps this package free of driver
 * dependencies but leaves every user writing the same closure. These helpers write it
 * for you, and each one is typed structurally rather than against the driver's own
 * types, so importing this module pulls in nothing.
 *
 * Every Postgres driver in JavaScript uses `$1` placeholders, so unlike the Python side
 * there is no style to get wrong here — but the helpers set it explicitly anyway, in
 * case a config arrives with `paramStyle: "pyformat"` copied from a Python example.
 *
 * ```ts
 * import { forPg } from "pghybrid/adapters";
 * const search = forPg(pool, { table: "chunks", textColumn: "content", vectorColumn: "embedding" });
 * ```
 */

import type { Config } from "./config.js";
import { HybridSearch } from "./search.js";
import type { Row } from "./search.js";

/** node-postgres: `Pool` and `Client` both satisfy this. */
export interface PgLike {
  query(text: string, values?: unknown[]): Promise<{ rows: Row[] }>;
}

/** postgres.js: the tagged-template function, which also carries `.unsafe`. */
export interface PostgresJsLike {
  unsafe(query: string, parameters?: unknown[]): PromiseLike<unknown>;
}

/** Drizzle exposes the driver it was constructed with as `$client`. */
export interface DrizzleLike {
  $client?: PgLike;
  execute?(query: unknown): Promise<unknown>;
}

/**
 * Kysely runs raw statements through `executeQuery` with a compiled query.
 *
 * The parameter is `any` on purpose. A `CompiledQuery` carries Kysely's full operation
 * node union plus an opaque `queryId`, and restating enough of that here to keep a real
 * `Kysely<T>` assignable would mean copying types that are not ours and will drift.
 * The shape actually passed is built below and is exercised against a live Kysely in
 * the test suite, which is a better guarantee than a hand-copied type.
 */
export interface KyselyLike {
   
  executeQuery(query: any): Promise<{ rows: Row[] }>;
}

export type Executor = (sql: string, params: unknown[]) => Promise<Row[]>;

function withNumericStyle(config: Config): Config {
  return { ...config, paramStyle: "numeric" };
}

function asRows(result: unknown): Row[] {
  if (Array.isArray(result)) return result as Row[];
  if (result && typeof result === "object" && Array.isArray((result as { rows?: unknown }).rows)) {
    return (result as { rows: Row[] }).rows;
  }
  throw new TypeError(
    "the driver returned something that is not a list of rows and has no .rows array; " +
      "pass your own execute callable instead",
  );
}

/** node-postgres (`pg`). Accepts a `Pool` or a `Client`. */
export function pgExecutor(client: PgLike): Executor {
  return async (sql, params) => (await client.query(sql, params)).rows;
}

export function forPg(client: PgLike, config: Config): HybridSearch {
  return new HybridSearch(withNumericStyle(config), pgExecutor(client));
}

/**
 * postgres.js.
 *
 * `unsafe` is the documented way to run a statement this library built, and it is not
 * a security hole here: the statement contains no interpolated values, only `$n`
 * placeholders, and every value still travels as a bound parameter.
 */
export function postgresJsExecutor(sql: PostgresJsLike): Executor {
  return async (query, params) => asRows(await sql.unsafe(query, params));
}

export function forPostgresJs(sql: PostgresJsLike, config: Config): HybridSearch {
  return new HybridSearch(withNumericStyle(config), postgresJsExecutor(sql));
}

/**
 * Drizzle.
 *
 * Goes through `$client`, the driver Drizzle was constructed with, because Drizzle's own
 * `execute` takes a query object rather than a statement and a parameter list — and
 * building one from raw SQL loses the bound parameters, which is the wrong trade.
 */
export function drizzleExecutor(db: DrizzleLike): Executor {
  const client = db.$client;
  if (!client || typeof client.query !== "function") {
    throw new TypeError(
      "this Drizzle instance exposes no $client to run raw SQL through. Pass the " +
        "underlying Pool to forPg instead, or supply your own execute callable.",
    );
  }
  return pgExecutor(client);
}

export function forDrizzle(db: DrizzleLike, config: Config): HybridSearch {
  return new HybridSearch(withNumericStyle(config), drizzleExecutor(db));
}

/** Kysely, via a raw compiled query so the parameters stay bound. */
export function kyselyExecutor(db: KyselyLike): Executor {
  return async (sql, params) => {
    // The same shape Kysely's own CompiledQuery.raw() produces: queryId is an opaque
    // marker it uses for plugin bookkeeping and carries nothing this needs.
    const compiled = {
      sql,
      parameters: params,
      query: { kind: "RawNode" },
      queryId: {},
    };
    return (await db.executeQuery(compiled)).rows;
  };
}

export function forKysely(db: KyselyLike, config: Config): HybridSearch {
  return new HybridSearch(withNumericStyle(config), kyselyExecutor(db));
}


// Prisma is deliberately absent. Its executor is one line —
//
//   (sql, params) => prisma.$queryRawUnsafe(sql, ...params)
//
// — and the README shows it, but nothing here ships without having been run against a
// real server, and the Prisma CLI in use while this was written could not generate a
// client to run it with. An adapter that has never executed is a claim, not a feature.
