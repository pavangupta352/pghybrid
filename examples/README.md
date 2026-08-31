# Examples

## `three_way_comparison.py`

Reproduces the table at the top of the main README: the same query run three ways
against the same corpus, printed side by side.

```bash
docker compose up -d
pip install "pghybrid[cli]"
python examples/three_way_comparison.py
```

It seeds twelve clauses from a software contract and asks **"renewal notice period"**.
The clause that answers the question says *"sixty days written notice prior to the
anniversary date"* and never uses the word *renewal*, so:

- **vector only** ranks a plausible-but-wrong clause first,
- **keyword only** ranks the clause that uses all three words and answers none of them first,
- **pghybrid** ranks the right one first.

The planted answer is second on both signals and first on neither, which is exactly the
case rank fusion exists to fix.

Embeddings are placed by hand on a circle rather than produced by a model. That means the
example needs no API key, no network and no GPU, and prints the same thing on every
machine, the geometry stands in for semantic similarity, while the keyword side is real
Postgres full-text search over real English.

## `langchain_retriever.py`

Uses pghybrid as a LangChain retriever, in about twenty lines.

```bash
docker compose up -d
pip install "pghybrid[cli]" langchain-core
python examples/langchain_retriever.py
```

The glue lives here rather than in the package, deliberately. LangChain's interfaces move
faster than this library needs to, and owning that compatibility would mean shipping a
dependency, a version matrix, and a release every time they cut one. Copy the class into
your project instead, it is yours to keep working, and CI runs this file so it cannot rot
unnoticed.

A retriever is handed a query *string*, while pghybrid wants the text and its embedding,
so the retriever owns the embedding step. That is also what lets the same index serve any
model: swap `DemoEmbeddings` for `OpenAIEmbeddings` and nothing else changes.

Every returned `Document` carries `matched_by`, `vector_rank` and `text_rank` in its
metadata, which signal found this row is usually the first thing you want to know when a
chain returns the wrong context, and a vector-only retriever cannot tell you.

LlamaIndex is the same shape: implement `_retrieve` and return `NodeWithScore` instead.
