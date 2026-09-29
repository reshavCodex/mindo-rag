from typing import Any, Dict, List

import cohere

from backend.app.config import settings


class CohereReranker:
    """
    Cloud-based reranker for MINDO RAG.

    Flow:

        Hybrid Retrieval
              ↓
        Top 20-30 candidates
              ↓
        Cohere Rerank API
              ↓
        Top 5-8 documents
    """

    def __init__(
        self,
        model: str = "rerank-v4.0-fast",
        top_k: int = 8,
    ) -> None:

        if not settings.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is not configured."
            )

        self.model = model
        self.top_k = top_k

        self.client = cohere.ClientV2(
            api_key=settings.cohere_api_key
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents against the query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return []

        query = query.strip()

        # Extract text for Cohere.
        document_texts = [
            document.get("text", "")
            for document in documents
        ]

        # Remove empty documents.
        valid_documents = [
            document
            for document, text in zip(
                documents,
                document_texts,
            )
            if text and text.strip()
        ]

        if not valid_documents:
            return []

        document_texts = [
            document["text"]
            for document in valid_documents
        ]

        try:

            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=document_texts,
                top_n=min(
                    self.top_k,
                    len(valid_documents),
                ),
            )

        except Exception as exc:

            raise RuntimeError(
                f"Cohere reranking failed: {exc}"
            ) from exc

        reranked_documents = []

        for result in response.results:

            original_document = valid_documents[
                result.index
            ]

            reranked_document = {
                **original_document,
                "rerank_score": float(
                    result.relevance_score
                ),
            }

            reranked_documents.append(
                reranked_document
            )

        return reranked_documents

    def close(self) -> None:
        """
        Close the Cohere client.

        The Cohere SDK manages HTTP resources internally.
        This method is provided for a consistent interface
        with the rest of the RAG components.
        """

        self.client = None

    def __enter__(
        self,
    ) -> "CohereReranker":

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

    documents = [
        {
            "chunk_id": "1",
            "text": (
                "Regular exercise can improve "
                "your mood and overall health."
            ),
            "metadata": {
                "source": "mental_health.pdf",
                "page": 1,
            },
        },
        {
            "chunk_id": "2",
            "text": (
                "Getting enough sleep is important "
                "for maintaining mental health."
            ),
            "metadata": {
                "source": "mental_health.pdf",
                "page": 2,
            },
        },
        {
            "chunk_id": "3",
            "text": (
                "The capital of France is Paris."
            ),
            "metadata": {
                "source": "other.pdf",
                "page": 1,
            },
        },
    ]

    with CohereReranker(
        top_k=3,
    ) as reranker:

        results = reranker.rerank(
            query=query,
            documents=documents,
        )

        print("\n==============================")
        print("COHERE RERANKER RESULTS")
        print("==============================")

        print(f"\nQuery: {query}")

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n--- Result {rank} ---"
            )

            print(
                f"Rerank Score: "
                f"{result['rerank_score']:.6f}"
            )

            print(
                f"Chunk ID: "
                f"{result['chunk_id']}"
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
                f"{result['text']}"
            )