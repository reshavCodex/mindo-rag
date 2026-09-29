from typing import Any, Dict, List, Optional

from backend.app.rag.embeddings import GeminiEmbeddingService
from backend.app.rag.vector_store import QdrantVectorStore


class RAGRetriever:
    """
    Dense vector retriever for MINDO RAG.

    Flow:

        Query
          ↓
        Gemini Embedding
          ↓
        Qdrant Similarity Search
          ↓
        Ranked Documents

    Current embedding configuration:

        Model:
            gemini-embedding-2

        Vector dimension:
            1536
    """

    EMBEDDING_MODEL = "gemini-embedding-2"
    VECTOR_SIZE = 1536

    def __init__(
        self,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.top_k = top_k
        self.score_threshold = score_threshold

        self.embedding_service = GeminiEmbeddingService(
            model=self.EMBEDDING_MODEL,
            output_dimension=self.VECTOR_SIZE,
        )

        self.vector_store = QdrantVectorStore(
            collection_name="mindo_knowledge",
            vector_size=self.VECTOR_SIZE,
        )

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents using
        dense vector similarity.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )

        return results

    def close(self) -> None:
        """
        Close the Qdrant connection.
        """

        if (
            getattr(
                self,
                "vector_store",
                None,
            )
            is not None
        ):

            self.vector_store.close()

            self.vector_store = None


if __name__ == "__main__":

    import time

    query = (
        "What are some ways to manage stress "
        "and improve sleep?"
    )

    print("\n==============================")
    print("MINDO DENSE RETRIEVER TEST")
    print("==============================")

    print(
        f"\nEmbedding model: "
        f"{RAGRetriever.EMBEDDING_MODEL}"
    )

    print(
        f"Vector dimension: "
        f"{RAGRetriever.VECTOR_SIZE}"
    )

    print(
        f"\nQuery:\n{query}"
    )

    retriever = None

    try:

        start_total = time.perf_counter()

        retriever = RAGRetriever(
            top_k=5
        )

        initialization_time = (
            time.perf_counter()
            - start_total
        )

        print(
            f"\nInitialization time: "
            f"{initialization_time:.2f} sec"
        )

        print(
            "\nRunning dense retrieval..."
        )

        start_retrieval = time.perf_counter()

        results = retriever.retrieve(
            query
        )

        retrieval_time = (
            time.perf_counter()
            - start_retrieval
        )

        print(
            f"\nRetrieval time: "
            f"{retrieval_time:.2f} sec"
        )

        print(
            f"Results returned: "
            f"{len(results)}"
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n--- Result {index} ---"
            )

            print(
                f"Score: "
                f"{result['score']:.4f}"
            )

            print(
                f"Source: "
                f"{result['metadata'].get('source')}"
            )

            print(
                f"Text: "
                f"{result['text'][:300]}"
            )

    finally:

        if retriever is not None:

            retriever.close()

            print(
                "\nRetriever closed."
            )

    print(
        "\n=============================="
    )