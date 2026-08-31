"""Use pghybrid as a LangChain retriever.

The glue is about twenty lines and it lives here rather than in the package. LangChain's
interfaces move faster than this library needs to, and owning that compatibility would
mean shipping a dependency, a version matrix and a release every time they cut one. Copy
the class below into your project instead; it is yours to keep working.

Run it:

    docker compose up -d
    pip install "pghybrid[cli]" langchain-core
    python examples/langchain_retriever.py

No API key and no network: the example embeds with a deterministic stand-in so it prints
the same thing everywhere. Swap `DemoEmbeddings` for `OpenAIEmbeddings`, `CohereEmbeddings`
or any other `Embeddings` implementation and nothing else changes, pghybrid never calls a
model, it takes the vector you hand it.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import psycopg
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from pghybrid import Config, HybridSearch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from seed_demo import DEMO_QUERY, PLANTED_TITLE  # noqa: E402
from seed_demo import main as seed

DSN = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")


class PgHybridRetriever(BaseRetriever):
    """A LangChain retriever backed by hybrid search on plain Postgres.

    The one thing worth noticing: a retriever is handed a query *string*, while pghybrid
    wants the query text and its embedding. So the retriever owns the embedding step, 
    which is also what lets the same index serve any model you point at it.
    """

    search: HybridSearch
    embeddings: Embeddings
    k: int = 4
    content_field: str = "content"

    # HybridSearch and Embeddings are not pydantic models, so the field types are taken
    # as-is rather than validated.
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        results = self.search.search(query, embedding=vector, limit=self.k)
        return [
            Document(
                page_content=str(result.get(self.content_field) or ""),
                metadata={
                    "id": result.id,
                    "score": result.score,
                    # Kept because they are the reason to use this retriever rather than
                    # a vector-only one: they say which signal found the row.
                    "vector_rank": result.vector_rank,
                    "text_rank": result.text_rank,
                    "matched_by": result.matched_by,
                    **{k: v for k, v in result.row.items() if k != self.content_field},
                },
            )
            for result in results
        ]


class DemoEmbeddings(Embeddings):
    """A stand-in model, so the example needs no key and gives the same answer everywhere.

    It reproduces the geometry the demo corpus was seeded with rather than pretending to
    understand English: the corpus places documents on a circle by hand, and this places
    the query at the origin of that circle.
    """

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 8
        vector[0] = math.cos(0.0)
        vector[1] = math.sin(0.0)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def build_retriever(connection: Any, k: int = 4) -> PgHybridRetriever:
    search = HybridSearch(
        Config(
            table="chunks",
            text_column="content",
            vector_column="embedding",
            tsvector_column="fts",
            extra_columns=["title"],
            paramstyle="pyformat",
        ),
        execute=lambda sql, params: connection.execute(sql, params).fetchall(),
    )
    return PgHybridRetriever(search=search, embeddings=DemoEmbeddings(), k=k)


def main() -> int:
    seed()
    with psycopg.connect(DSN, row_factory=psycopg.rows.dict_row) as connection:
        retriever = build_retriever(connection)
        documents = retriever.invoke(DEMO_QUERY)

        print(f"\n  retriever.invoke({DEMO_QUERY!r}) returned {len(documents)} documents\n")
        for position, document in enumerate(documents, 1):
            metadata = document.metadata
            marker = "  <- the clause that answers it" if metadata["title"] == PLANTED_TITLE else ""
            print(
                f"  {position}. {metadata['title']:<30} "
                f"score {metadata['score']:.6f}  "
                f"[{metadata['matched_by']}]{marker}"
            )
            print(f"     {document.page_content[:88]}")

    if not documents or documents[0].metadata["title"] != PLANTED_TITLE:
        print("\n  the retriever did not return the planted answer first", file=sys.stderr)
        return 1

    print(
        "\n  Every document carries which signal found it, which a vector-only\n"
        "  retriever cannot tell you, and is usually the first thing you want\n"
        "  when a chain returns the wrong context.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
