import time
from typing import Any, Dict, List

from backend.app.rag.hybrid_retriever import HybridRetriever
from backend.app.rag.reranker import CohereReranker


class RAGRetrievalPipeline:
    """
    MINDO retrieval pipeline.

    Flow:

        Query
          ↓
        Hybrid Retrieval
          ├── Dense Retrieval
          └── BM25 Sparse Retrieval
          ↓
        Reciprocal Rank Fusion
          ↓
        Cohere Reranking
          ↓
        Standardized Evidence
    """

    def __init__(
        self,
        candidate_k: int = 20,
        final_k: int = 5,
        reranker_model: str = "rerank-v4.0-fast",
    ) -> None:

        self.candidate_k = candidate_k
        self.final_k = final_k

        self.hybrid_retriever = HybridRetriever(
            dense_top_k=candidate_k,
            sparse_top_k=candidate_k,
            final_top_k=candidate_k,
        )

        self.reranker = CohereReranker(
            model=reranker_model,
            top_k=final_k,
        )

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute the complete retrieval pipeline.

        Returns standardized evidence containing:
        - text
        - rerank score
        - source
        - page information
        - original metadata
        - retrieval scores
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # ---------------------------------------------------------
        # 1. Hybrid retrieval
        # ---------------------------------------------------------

        candidates = self.hybrid_retriever.retrieve(
            query=query
        )

        if not candidates:
            return []

        # ---------------------------------------------------------
        # 2. Cohere reranking
        # ---------------------------------------------------------

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
        )

        if not reranked:
            return []

        # ---------------------------------------------------------
        # 3. Standardize evidence
        # ---------------------------------------------------------

        evidence: List[Dict[str, Any]] = []

        for document in reranked:

            metadata = document.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            # -----------------------------------------------------
            # Page metadata
            #
            # The ingestion pipeline stores:
            #     page_start
            #     page_end
            #
            # Preserve those values explicitly in the evidence
            # object so downstream components do not need to inspect
            # nested metadata.
            # -----------------------------------------------------

            page_start = metadata.get(
                "page_start"
            )

            page_end = metadata.get(
                "page_end"
            )

            evidence.append(
                {
                    "text": document.get(
                        "text",
                        "",
                    ),
                    "score": float(
                        document.get(
                            "rerank_score",
                            0.0,
                        )
                    ),
                    "source": metadata.get(
                        "source"
                    ),

                    # Explicit source location
                    "page_start": page_start,
                    "page_end": page_end,

                    # Preserve complete metadata
                    "metadata": metadata,

                    # Retrieval diagnostics
                    "retrieval": {
                        "dense_score": document.get(
                            "dense_score"
                        ),
                        "sparse_score": document.get(
                            "sparse_score"
                        ),
                        "rrf_score": document.get(
                            "rrf_score"
                        ),
                        "reranker": "cohere",
                        "reranker_model": self.reranker.model,
                    },
                }
            )

        return evidence

    def close(self) -> None:
        """
        Close the underlying retrieval resources.
        """

        if (
            getattr(
                self,
                "hybrid_retriever",
                None,
            )
            is not None
        ):

            self.hybrid_retriever.close()

            self.hybrid_retriever = None


if __name__ == "__main__":

    query = (
        "What are some ways to manage stress "
        "and improve sleep?"
    )

    print("\n==============================")
    print("MINDO RETRIEVAL PIPELINE TEST")
    print("==============================")

    print(
        f"\nQuery:\n{query}"
    )

    pipeline = None

    try:

        start_total = time.perf_counter()

        print(
            "\nInitializing retrieval pipeline..."
        )

        pipeline = RAGRetrievalPipeline(
            candidate_k=20,
            final_k=5,
        )

        initialization_time = (
            time.perf_counter()
            - start_total
        )

        print(
            f"Initialization time: "
            f"{initialization_time:.2f} sec"
        )

        print(
            "\nRunning retrieval..."
        )

        start_retrieval = time.perf_counter()

        evidence = pipeline.retrieve(
            query=query
        )

        retrieval_time = (
            time.perf_counter()
            - start_retrieval
        )

        total_time = (
            time.perf_counter()
            - start_total
        )

        print(
            f"\nRetrieval time: "
            f"{retrieval_time:.2f} sec"
        )

        print(
            f"Total test time: "
            f"{total_time:.2f} sec"
        )

        print(
            f"\nFinal evidence: "
            f"{len(evidence)}"
        )

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            print(
                f"\n--- Evidence {index} ---"
            )

            print(
                f"Score: "
                f"{item['score']:.4f}"
            )

            print(
                f"Source: "
                f"{item['source']}"
            )

            print(
                f"Page: "
                f"{item.get('page_start')} - "
                f"{item.get('page_end')}"
            )

            print(
                f"Text: "
                f"{item['text'][:300]}"
            )

            retrieval_info = item.get(
                "retrieval",
                {},
            )

            print(
                f"Dense score: "
                f"{retrieval_info.get('dense_score')}"
            )

            print(
                f"Sparse score: "
                f"{retrieval_info.get('sparse_score')}"
            )

            print(
                f"RRF score: "
                f"{retrieval_info.get('rrf_score')}"
            )

    finally:

        if pipeline is not None:

            pipeline.close()

            print(
                "\nRetrieval resources closed."
            )

    print(
        "\n=============================="
    )
    print(
        "MINDO RETRIEVAL PIPELINE TEST COMPLETE"
    )
    print(
        "=============================="
    )