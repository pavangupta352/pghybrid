# pghybrid

**Hybrid search on the Postgres you already have.**

Vector similarity + full-text search, combined by Reciprocal Rank Fusion, on plain
`pgvector`. No `pg_search`, no VectorChord, no Elasticsearch, no vector database, no
extension you need superuser to install.

```bash
npm install pghybrid
```

Zero runtime dependencies. `pghybrid` generates SQL and hands it to the driver you
already use, node-postgres, postgres.js, Drizzle, Supabase. It never opens a connection
of its own and never calls an embedding provider.

<!-- check:ts -->
```ts
import { HybridSearch } from "pghybrid";

const search = new HybridSearch(
  { table: "chunks", textColumn: "content", vectorColumn: "embedding", tsvectorColumn: "fts" },
  (sql, params) => pool.query(sql, params).then((result) => result.rows),
);

const rows = await search.search("renewal notice period", { embedding, limit: 10 });
for (const row of rows) {
  console.log(row.score, row.matchedBy, row.row.title);
}
```

Read the statement instead of running it:

```ts
const { sql, params } = search.buildQuery("renewal notice period", { embedding, limit: 10 });
```

This package is the TypeScript half of [pghybrid][repo]. It generates byte-identical SQL
to the Python package, which is checked on every commit rather than assumed.

**Full documentation, the reasoning, and the copy-and-paste SQL: [github.com/pavangupta352/pghybrid][repo].**

MIT © Pavan Gupta

[repo]: https://github.com/pavangupta352/pghybrid#readme
