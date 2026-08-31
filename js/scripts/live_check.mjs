#!/usr/bin/env node
/**
 * Run one real query against a real database.
 *
 * The unit suite proves the package generates the right string. It cannot prove the
 * string is valid SQL, that the placeholders line up with what the driver sends, or
 * that the vector literal is a shape pgvector accepts — all of which are ways a query
 * generator passes its own tests and fails on a server.
 *
 * The corpus is the demo one from `scripts/seed_demo.py`. The row this asserts on is
 * second on both signals and first on neither, so it comes back first only if the
 * fusion actually happened.
 *
 *     python/.venv/bin/python scripts/seed_demo.py
 *     npm run build --prefix js && node js/scripts/live_check.mjs
 */

import pg from "pg";

import { HybridSearch } from "../dist/index.js";

const DSN =
  process.env.PGHYBRID_TEST_DSN ?? "postgresql://postgres:pghybrid@localhost:55432/pghybrid";

const EXPECTED_TITLE = "Termination for convenience";
const EXPECTED_SCORE = 0.032258;

const client = new pg.Client({ connectionString: DSN });
await client.connect();

try {
  const search = new HybridSearch(
    {
      table: "chunks",
      textColumn: "content",
      vectorColumn: "embedding",
      idColumn: "id",
      tsvectorColumn: "fts",
      extraColumns: ["title"],
      filterColumns: ["tenant_id"],
      paramStyle: "numeric",
    },
    (sql, params) => client.query(sql, params).then((result) => result.rows),
  );

  const results = await search.search("renewal notice period", {
    embedding: [1, 0, 0, 0, 0, 0, 0, 0],
    limit: 5,
  });

  for (const [index, result] of results.entries()) {
    process.stdout.write(
      `  ${index + 1}  ${result.score.toFixed(6)}  ` +
        `vector ${String(result.vectorRank ?? "—").padStart(2)}  ` +
        `text ${String(result.textRank ?? "—").padStart(2)}  ` +
        `${String(result.row.title)}\n`,
    );
  }

  const top = results[0];
  if (top === undefined) {
    throw new Error("the query returned no rows; run scripts/seed_demo.py first");
  }
  if (top.row.title !== EXPECTED_TITLE) {
    throw new Error(
      `expected ${JSON.stringify(EXPECTED_TITLE)} first, got ` +
        `${JSON.stringify(top.row.title)}. Neither signal ranks it first, so this is ` +
        "what a broken fusion looks like.",
    );
  }
  if (Math.abs(top.score - EXPECTED_SCORE) > 1e-6) {
    throw new Error(`expected a score of ${EXPECTED_SCORE}, got ${top.score}`);
  }

  process.stdout.write(`\n${EXPECTED_TITLE} ranked first, on rank 2 of both signals.\n`);
} finally {
  await client.end();
}
