import pg from "pg";
import { HybridSearch } from "/Users/pavan/dev/Postgres hybrid search/js/dist/index.js";
const client = new pg.Client({ connectionString: "postgresql://postgres:pghybrid@localhost:55432/pghybrid" });
await client.connect();
const search = new HybridSearch(
  { table: "chunks", textColumn: "content", vectorColumn: "embedding", idColumn: "id",
    tsvectorColumn: "fts", extraColumns: ["title"], filterColumns: ["tenant_id"], paramStyle: "numeric" },
  async (sql, params) => { console.log(JSON.stringify(params)); const r = await client.query(sql, params); console.log(r.fields.map(f=>`${f.name}:${f.dataTypeID}`).join(" ")); console.log(r.rows.slice(0,3)); return r.rows; },
);
await search.search("renewal notice period", { embedding: [1,0,0,0,0,0,0,0], limit: 5 });
await client.end();
