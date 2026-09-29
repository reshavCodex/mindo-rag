from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Sparse keyword-based retrieval using BM25.

    BM25 is useful for:
    - Exact terminology
    - Medical/assessment keywords
    - Names of conditions
    - Specific phrases
    - Rare words that dense retrieval may miss
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        self.documents = documents

        # Tokenize document text
        tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Basic normalization and tokenization.
        """

        return (
            text.lower()
            .strip()
            .split()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 relevance scoring.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            document = self.documents[index]

            results.append(
                {
                    "chunk_id": document.get(
                        "chunk_id"
                    ),
                    "score": float(
                        scores[index]
                    ),
                    "text": document.get(
                        "text",
                        "",
                    ),
                    "metadata": document.get(
                        "metadata",
                        {},
                    ),
                }
            )

        return results


if __name__ == "__main__":

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
                "Staying connected with friends "
                "and family can provide support."
            ),
            "metadata": {
                "source": "mental_health.pdf",
                "page": 3,
            },
        },
    ]

    retriever = BM25Retriever(
        documents=documents
    )

    query = "mental health sleep"

    results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    print("\n==============================")
    print("BM25 RETRIEVAL RESULTS")
    print("==============================")

    print(f"\nQuery: {query}")

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
            f"Text: "
            f"{result['text']}"
        )

        print(
            f"Metadata: "
            f"{result['metadata']}"
        )