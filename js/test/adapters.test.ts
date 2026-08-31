/**
 * The adapters, against a real server.
 *
 * A driver is a transport, not a dialect: node-postgres, postgres.js, Drizzle and
 * Kysely must all return the same rows in the same order with the same scores. These
 * are the only tests in this package that need a database, and they skip when none is
 * reachable so `npm test` stays useful on a laptop with nothing running.
 *
 * The fixture is created here rather than borrowed from the Python seed script, so the
 * TypeScript suite has no dependency on a Python toolchain being present.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  forDrizzle,
  forKysely,
  forPg,
  forPostgresJs,
} from "../src/adapters.js";
import type { Config } from "../src/config.js";

const DSN =
  process.env.PGHYBRID_TEST_DSN ??
  "postgresql://postgres:pghybrid@localhost:55432/pghybrid";

const TABLE = "adapter_fixture";
const QUERY = "renewal notice period";
const EMBEDDING = [1, 0, 0, 0, 0, 0, 0, 0];
const ANSWER = "Termination for convenience";

const CONFIG: Config = {
  table: TABLE,
  textColumn: "content",
  vectorColumn: "embedding",
  tsvectorColumn: "fts",
  extraColumns: ["title"],
};

/** Angle from the query vector, title, body. Mirrors scripts/seed_demo.py. */
const DOCUMENTS: [number, string, string][] = [
  [
    0.15,
    "Automatic extension",
    "This agreement extends automatically for successive twelve month terms unless either party elects otherwise. Extension begins on the anniversary of the effective date.",
  ],
  [
    0.28,
    ANSWER,
    "Either party may terminate this agreement for convenience by giving sixty days written notice prior to the anniversary date. The notice period runs from the date of delivery.",
  ],
  [
    0.4,
    "Subscription term",
    "The initial subscription term is twelve months from the effective date and continues until terminated in accordance with this section.",
  ],
  [
    0.52,
    "Fees and invoicing",
    "Fees are invoiced annually in advance. Invoices are payable within thirty days of the invoice date.",
  ],
  [
    0.64,
    "Service levels",
    "The supplier will use commercially reasonable efforts to maintain a monthly uptime percentage of at least 99.9 percent.",
  ],
  [
    0.76,
    "Notice requirements",
    "Any notice given under this agreement must be in writing and delivered to the address set out in the order form.",
  ],
  [
    0.88,
    "Renewal pricing",
    "Renewal pricing is subject to change on notice. The supplier will notify the customer before the renewal period commences, and any renewal notice must state the revised fees.",
  ],
  [
    1.0,
    "Renewal terms",
    "Renewal terms and conditions apply to all customers on the standard plan from the start of each renewal period.",
  ],
  [
    1.12,
    "Governing law",
    "This agreement is governed by the laws of England and Wales and the parties submit to the exclusive jurisdiction of its courts.",
  ],
  [
    1.24,
    "Confidentiality",
    "Each party shall keep confidential all information disclosed by the other party and shall not disclose it to any third party.",
  ],
  [
    1.36,
    "Data protection",
    "The supplier processes personal data only on documented instructions from the customer and in accordance with applicable data protection law.",
  ],
  [
    1.48,
    "Limitation of liability",
    "Neither party is liable for indirect or consequential loss arising out of or in connection with this agreement.",
  ],
];

function unitVector(angle: number): string {
  const v = new Array(8).fill(0);
  v[0] = Math.cos(angle);
  v[1] = Math.sin(angle);
  return `[${v.join(",")}]`;
}

let pool: any;

let reachable = false;

beforeAll(async () => {
  try {
    const pg = (await import("pg")).default;
    pool = new pg.Pool({
      connectionString: DSN,
      connectionTimeoutMillis: 3000,
    });
    await pool.query("SELECT 1");
    reachable = true;
  } catch {
    return;
  }

  await pool.query("CREATE EXTENSION IF NOT EXISTS vector");
  await pool.query(`DROP TABLE IF EXISTS ${TABLE} CASCADE`);
  await pool.query(`
    CREATE TABLE ${TABLE} (
      id bigserial PRIMARY KEY,
      title text NOT NULL,
      content text NOT NULL,
      embedding vector(8),
      fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
    )`);
  for (const [angle, title, content] of DOCUMENTS) {
    await pool.query(
      `INSERT INTO ${TABLE} (title, content, embedding) VALUES ($1, $2, $3)`,
      [title, content, unitVector(angle)],
    );
  }
}, 30_000);

afterAll(async () => {
  if (!pool) return;
  if (reachable) await pool.query(`DROP TABLE IF EXISTS ${TABLE} CASCADE`);
  await pool.end();
});

describe("adapters against a live server", () => {
  const expected = [ANSWER, "Renewal pricing", "Renewal terms"];

  /**
   * Skip at run time rather than with `describe.skipIf`.
   *
   * Reachability is only known after `beforeAll` has tried to connect, and gating on an
   * environment variable instead meant these silently skipped for anyone who followed
   * CONTRIBUTING — `docker compose up -d` then `npm test` — because the database was up
   * but PGHYBRID_TEST_DSN was not set. The Python suite needs no such variable, and
   * neither should this. Reporting them as skipped, rather than returning early and
   * passing, keeps the count honest.
   */
  function requireDatabase(context: { skip: () => void }): void {
    if (!reachable) context.skip();
  }

  it("node-postgres returns the fused ranking", async (context) => {
    requireDatabase(context);
    const rows = await forPg(pool, CONFIG).search(QUERY, {
      embedding: EMBEDDING,
      limit: 3,
    });
    expect(rows.map((r) => String(r.row.title))).toEqual(expected);
    // Every score above zero: the ::float8 casts in the fusion are what keep this
    // true when a driver sends the weights as integers.
    expect(rows.every((r) => r.score > 0)).toBe(true);
  });

  it("postgres.js agrees with node-postgres", async (context) => {
    requireDatabase(context);
    const postgres = (await import("postgres")).default;
    const sql = postgres(DSN);
    try {
      const rows = await forPostgresJs(sql, CONFIG).search(QUERY, {
        embedding: EMBEDDING,
        limit: 3,
      });
      expect(rows.map((r) => String(r.row.title))).toEqual(expected);
    } finally {
      await sql.end();
    }
  });

  it("drizzle agrees, through its underlying client", async (context) => {
    requireDatabase(context);
    const { drizzle } = await import("drizzle-orm/node-postgres");
    const rows = await forDrizzle(drizzle(pool), CONFIG).search(QUERY, {
      embedding: EMBEDDING,
      limit: 3,
    });
    expect(rows.map((r) => String(r.row.title))).toEqual(expected);
  });

  it("kysely agrees, through a raw compiled query", async (context) => {
    requireDatabase(context);
    const { Kysely, PostgresDialect } = await import("kysely");
    const pg = (await import("pg")).default;
    // Kysely's destroy() ends the pool it was handed, so it gets its own rather than
    // closing the one the rest of the suite is still using.
    const ownPool = new pg.Pool({ connectionString: DSN });

    const db = new Kysely<any>({
      dialect: new PostgresDialect({ pool: ownPool }),
    });
    try {
      const rows = await forKysely(db, CONFIG).search(QUERY, {
        embedding: EMBEDDING,
        limit: 3,
      });
      expect(rows.map((r) => String(r.row.title))).toEqual(expected);
    } finally {
      await db.destroy();
    }
  });

  it("forces the numeric placeholder style whatever the config asked for", async (context) => {
    requireDatabase(context);
    // Every Postgres driver in JavaScript speaks $1; a pyformat config copied from a
    // Python example would otherwise produce %s and fail on parameter counts.
    const search = forPg(pool, { ...CONFIG, paramStyle: "pyformat" });
    const { sql } = search.buildQuery(QUERY, {
      embedding: EMBEDDING,
      limit: 3,
    });
    expect(sql).toContain("$1");
    expect(sql).not.toContain("%s");
  });
});
