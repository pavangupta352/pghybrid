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
machine — the geometry stands in for semantic similarity, while the keyword side is real
Postgres full-text search over real English.
