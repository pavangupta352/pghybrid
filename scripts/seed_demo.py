"""Seed the demonstration corpus used by the README and the integration tests.

The corpus is a small set of contract clauses plus one planted answer. Embeddings are
placed by hand on a circle rather than produced by a model, for three reasons: the
fixture stays byte-identical across machines, the tests need no API key and no
network, and the vector ranking becomes something the test can state exactly instead
of hoping for.

The planted answer is deliberately mid-ranked by *both* signals. Neither vector-only
nor keyword-only search returns it first; Reciprocal Rank Fusion does. That is the
property the golden test asserts, and it is the entire argument for hybrid search.
"""

from __future__ import annotations

import math
import os
import sys

import psycopg

DIMENSIONS = 8

# (angle_from_query, title, content)
#
# The angle controls cosine distance and therefore the vector ranking: smaller angle
# means a closer match. The text ranking falls out of real ts_rank_cd over real
# English, which is why the wording matters as much as the geometry.
DOCUMENTS: list[tuple[float, str, str]] = [
    # Closest vector, but contains none of the query's words: the top semantic hit is
    # invisible to keyword search. This is the row that makes vector-only look right
    # and keyword-only look broken.
    (
        0.15,
        "Automatic extension",
        "This agreement extends automatically for successive twelve month terms "
        "unless either party elects otherwise. Extension begins on the anniversary "
        "of the effective date.",
    ),
    # THE PLANTED ANSWER. Second on both signals, first on neither.
    (
        0.28,
        "Termination for convenience",
        "Either party may terminate this agreement for convenience by giving sixty "
        "days written notice prior to the anniversary date. The notice period runs "
        "from the date of delivery.",
    ),
    (
        0.40,
        "Subscription term",
        "The initial subscription term is twelve months from the effective date and "
        "continues until terminated in accordance with this section.",
    ),
    (
        0.52,
        "Fees and invoicing",
        "Fees are invoiced annually in advance. Invoices are payable within thirty "
        "days of the invoice date.",
    ),
    (
        0.64,
        "Service levels",
        "The supplier will use commercially reasonable efforts to maintain a monthly "
        "uptime percentage of at least 99.9 percent.",
    ),
    (
        0.76,
        "Notice requirements",
        "Any notice given under this agreement must be in writing and delivered to "
        "the address set out in the order form.",
    ),
    # Strongest keyword match, weakest semantics: it uses every word in the query and
    # answers none of it. This is the row that makes keyword-only look right.
    (
        0.88,
        "Renewal pricing",
        "Renewal pricing is subject to change on notice. The supplier will notify the "
        "customer before the renewal period commences, and any renewal notice must "
        "state the revised fees.",
    ),
    (
        1.00,
        "Renewal terms",
        "Renewal terms and conditions apply to all customers on the standard plan "
        "from the start of each renewal period.",
    ),
    (
        1.12,
        "Governing law",
        "This agreement is governed by the laws of England and Wales and the parties "
        "submit to the exclusive jurisdiction of its courts.",
    ),
    (
        1.24,
        "Confidentiality",
        "Each party shall keep confidential all information disclosed by the other "
        "party and shall not disclose it to any third party.",
    ),
    (
        1.36,
        "Data protection",
        "The supplier processes personal data only on documented instructions from "
        "the customer and in accordance with applicable data protection law.",
    ),
    (
        1.48,
        "Limitation of liability",
        "Neither party is liable for indirect or consequential loss arising out of "
        "or in connection with this agreement.",
    ),
]

# The clause a person is actually looking for when they type the demo query. It is
# the answer because it states the number of days; it is hard to retrieve because it
# never uses the word "renewal".
PLANTED_TITLE = "Termination for convenience"
DEMO_QUERY = "renewal notice period"


def unit_vector(angle: float) -> list[float]:
    """A unit vector at ``angle`` radians from the query vector [1, 0, 0, ...]."""
    vector = [0.0] * DIMENSIONS
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return vector


def query_vector() -> list[float]:
    return unit_vector(0.0)


def to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


SCHEMA = """
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id          bigserial PRIMARY KEY,
    tenant_id   integer NOT NULL DEFAULT 1,
    title       text NOT NULL,
    content     text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    embedding   vector(8),
    fts         tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
);
CREATE INDEX chunks_fts_idx ON chunks USING gin (fts);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def main() -> int:
    dsn = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA)
        for angle, title, content in DOCUMENTS:
            conn.execute(
                "INSERT INTO chunks (title, content, embedding) VALUES (%s, %s, %s)",
                (title, content, to_pgvector(unit_vector(angle))),
            )
        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"seeded {count} rows into chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
