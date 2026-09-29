from typing import Any, Dict, List, Optional

from backend.app.rag.retriever import RAGRetriever
from backend.app.rag.bm25_retriever import BM25Retriever
from backend.app.rag.ingestion.loader import load_knowledge_base
from backend.app.rag.ingestion.chunker import chunk_documents
from backend.app.rag.ingestion.metadata import enrich_documents


class HybridRetriever:
    """
    Hybrid retrieval for MINDO.

    Combines:

        Dense Retrieval
            ↓
        Qdrant / Gemini Embeddings

        Sparse Retrieval
            ↓
        BM25

        Both rankings
            ↓
        Reciprocal Rank Fusion (RRF)
            ↓
        Unified ranked results
    """

    def __init__(
        self,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        final_top_k: int = 10,
        rrf_k: int = 60,
        score_threshold: Optional[float] = None,
    ) -> None:

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

        # Dense retriever
        self.dense_retriever = RAGRetriever(
            top_k=dense_top_k,
            score_threshold=score_threshold,
        )

        # Build BM25 corpus from the knowledge base
        documents = load_knowledge_base()

        if not documents:
            raise RuntimeError(
                "No documents found in the knowledge base."
            )

        chunks = chunk_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200,
        )

        enriched_chunks = enrich_documents(chunks)

        self.sparse_retriever = BM25Retriever(
            documents=enriched_chunks
        )

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval using dense + sparse search.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # --------------------------------------------------
        # 1. Dense retrieval
        # --------------------------------------------------

        dense_results = self.dense_retriever.retrieve(
            query=query
        )

        # --------------------------------------------------
        # 2. Sparse retrieval
        # --------------------------------------------------

        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            top_k=self.sparse_top_k,
        )

        # --------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # --------------------------------------------------

        fused_results = {}

        self._add_results(
            fused_results,
            dense_results,
            source="dense",
        )

        self._add_results(
            fused_results,
            sparse_results,
            source="sparse",
        )

        # --------------------------------------------------
        # 4. Sort by RRF score
        # --------------------------------------------------

        ranked_results = sorted(
            fused_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        # --------------------------------------------------
        # 5. Return final results
        # --------------------------------------------------

        return ranked_results[:self.final_top_k]

    def _add_results(
        self,
        fused_results: Dict[str, Dict[str, Any]],
        results: List[Dict[str, Any]],
        source: str,
    ) -> None:
        """
        Add retrieval results into the RRF ranking.
        """

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            # RRF formula:
            #
            # RRF Score = 1 / (k + rank)
            #
            rrf_score = 1.0 / (
                self.rrf_k + rank
            )

            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "rrf_score": 0.0,
                    "retrieval_sources": [],
                    "dense_score": None,
                    "sparse_score": None,
                }

            fused_results[chunk_id][
                "rrf_score"
            ] += rrf_score

            if source not in fused_results[
                chunk_id
            ]["retrieval_sources"]:

                fused_results[
                    chunk_id
                ]["retrieval_sources"].append(
                    source
                )

            if source == "dense":

                fused_results[
                    chunk_id
                ]["dense_score"] = result.get(
                    "score"
                )

            elif source == "sparse":

                fused_results[
                    chunk_id
                ]["sparse_score"] = result.get(
                    "score"
                )

    def close(self) -> None:
        """Close the dense retriever resources."""

        self.dense_retriever.close()

    def __enter__(self) -> "HybridRetriever":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.close()


if __name__ == "__main__":

    query = (
        "What are some ways to take care "
        "of your mental health?"
    )

    with HybridRetriever(
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=5,
    ) as retriever:

        results = retriever.retrieve(
            query=query
        )

        print("\n==============================")
        print("HYBRID RETRIEVAL RESULTS")
        print("==============================")

        print(f"\nQuery: {query}")
        print(
            f"Results found: {len(results)}"
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n--- Result {index} ---"
            )

            print(
                f"RRF Score: "
                f"{result['rrf_score']:.6f}"
            )

            print(
                f"Dense Score: "
                f"{result['dense_score']}"
            )

            print(
                f"Sparse Score: "
                f"{result['sparse_score']}"
            )

            print(
                f"Sources: "
                f"{result['retrieval_sources']}"
            )

            print(
                f"Source: "
                f"{result['metadata'].get('source')}"
            )

            print(
                f"Page: "
                f"{result['metadata'].get('page')}"
            )

            print(
                f"Text: "
                f"{result['text'][:500]}"
            )