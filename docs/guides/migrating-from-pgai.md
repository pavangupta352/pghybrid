# pgai is archived. Here is what still works, and what you have to replace.

[`timescale/pgai`](https://github.com/timescale/pgai) is archived. Its README says it
plainly: *"As of February 2026, this project is no longer being maintained or supported."*
5,800 stars, and no more releases.

If you are running it, nothing broke today. This is about what to do before something does.

## First, the good news: your data is ordinary Postgres

pgai's `ai.destination_table()` created two objects — a store table holding the chunks and
embeddings, and a view joining that store back to your source table:

```sql
ai.create_vectorizer(
    'blog'::regclass,
    loading     => ai.loading_column('contents'),
    embedding   => ai.embedding_ollama('nomic-embed-text', 768),
    destination => ai.destination_table('blog_contents_embeddings')
);
```

Both are plain relations. A table of `vector` columns and a view over it keep working
whether or not anything maintains them, and they survive the extension being uninstalled.
What you lose is the automation around them, not the embeddings themselves.

So this is not a data migration. It is a question of which of pgai's jobs you still need
doing, and by what.

## Be clear about what pghybrid does and does not replace

pgai did two quite different things.

| pgai did | pghybrid |
|---|---|
| **Embedded your rows automatically** — a worker watched the source table, chunked, called the model, kept the store fresh | ❌ **No.** Not even slightly. |
| **Embedded your query** at search time via `ai.ollama_embed(...)` / `ai.openai_embed(...)` | ❌ No. You pass a vector in. |
| **Chunked** your text | ❌ No. |
| **Ran the search** over the resulting embeddings | ✅ Yes, and adds keyword search fused with it. |

That first row is the important one. The vectorizer is genuinely useful and pghybrid is not
a substitute for it. If automatic re-embedding on write is what you valued, your options
are Timescale's hosted service, running the archived worker until it stops building, or
moving embedding into your application. Do not let this page talk you out of a feature you
actually use.

What follows assumes you have decided to own the embedding step yourself, which most people
reaching for a replacement already have.

## Point pghybrid at the objects you already have

```bash
pip install pghybrid          # or: npm install pghybrid
pghybrid init --dsn "$DATABASE_URL" --table blog_contents_embeddings
```

`init` reads the store table, finds the `chunk` and `embedding` columns on its own, and
prints the migration — a stored `tsvector` for the keyword half, a GIN index on it, and an
HNSW index sized for the row count you actually have. Nothing else changes.

The view works too:

```bash
pghybrid search "renewal notice period" --table blog_contents_embeddings --embedding-from 1
```

A view can be searched but cannot carry an index, so `init` against one prints a note
pointing at the store table underneath rather than DDL Postgres would reject. Search
against whichever is more convenient; index the table.

## Replacing the search

The pgai query embedded the text inline and ordered by distance:

```sql
-- before: needs the ai extension and a reachable model
SELECT chunk,
       embedding <=> ai.ollama_embed('nomic-embed-text', 'renewal notice period',
                                     host => 'http://ollama:11434') AS distance
FROM blog_contents_embeddings
ORDER BY distance
LIMIT 10;
```

The `ai.*_embed` call is the part that goes away. Embed the query in your application and
pass the vector:

<!-- check:python -->
```python
from pghybrid import Config, HybridSearch

search = HybridSearch(
    Config(table="blog_contents_embeddings", text_column="chunk",
           vector_column="embedding", tsvector_column="fts", paramstyle="pyformat"),
    execute=lambda sql, params: conn.execute(sql, params).fetchall(),
)

vector = embed("renewal notice period")          # whatever you already use
for row in search.search("renewal notice period", embedding=vector, limit=10):
    print(row.score, row.get("chunk"))
```

Two things you gain by doing it this way, neither of which pgai offered:

- **The keyword signal.** The same query text is also run through Postgres full-text
  search and fused with the vector ranking, so a chunk that uses your exact term is found
  even when the embedding does not place it nearby. That is the whole point of the
  library; the [three-way comparison](../../README.md) shows a case where neither signal
  alone returns the right row.
- **The model is yours.** No extension, no worker, no host to reach from inside the
  database. It runs on managed Postgres where the `ai` extension was never installable.

## Do not sum the two scores

Whatever you build, resist writing this:

```sql
0.7 * (1 - (embedding <=> $1)) + 0.3 * ts_rank_cd(fts, query)
```

It reads as *70% semantic, 30% keyword* and is not. Cosine distance is bounded in `[0,1]`
and clusters tightly; `ts_rank_cd` is unbounded and small. The weights describe the
constants, not the influence, and tuning them does not fix it because the spans are set by
the scoring functions. Fusing ranks avoids the problem, because ranks share a scale by
construction. `pghybrid explain` measures the difference on your own data.

## Then check the indexes pgai was choosing for you

`ai.indexing_diskann()` and `ai.indexing_hnsw()` picked and maintained your vector index.
Nothing does that now, so it is worth looking at what you were left with:

```bash
pghybrid doctor --dsn "$DATABASE_URL" --table blog_contents_embeddings
```

That reports recall@k measured against exact search, whether the index is being used under
your filters, and what to change. An under-tuned vector index does not error — it quietly
returns worse answers — which is exactly the failure mode to check for after the thing that
was tuning it stops being maintained.

## If you keep the vectorizer

Perfectly reasonable, and the two are not exclusive: pgai keeps the store table fresh,
pghybrid searches it. You gain keyword search and lose nothing. When the worker eventually
stops building against a current Postgres, the search half is already migrated.

---

Corrections and questions: <https://github.com/pavangupta352/pghybrid/issues>
