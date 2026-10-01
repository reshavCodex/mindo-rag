import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models


# Load environment variables from .env when running locally.
# On Render, these values will come from the service environment.
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STORAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "qdrant"
)

DEFAULT_COLLECTION_NAME = "mindo_knowledge"
DEFAULT_VECTOR_SIZE = 1536


class QdrantVectorStore:
    """
    Persistent Qdrant vector store for MINDO RAG.

    Supports:

        - Qdrant Cloud
        - Local persistent Qdrant
        - Vector upsert
        - Similarity search
        - Collection management
        - Clean resource lifecycle

    Connection priority:

        1. QDRANT_URL + QDRANT_API_KEY
           → Qdrant Cloud / remote Qdrant

        2. Local storage path
           → Local persistent Qdrant

    This allows Render to use Qdrant Cloud while preserving
    local Qdrant as a development fallback.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        storage_path: Optional[str | Path] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:

        self.collection_name = collection_name
        self.vector_size = vector_size

        if self.vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than zero."
            )

        # --------------------------------------------------
        # Resolve Qdrant configuration
        # --------------------------------------------------

        # Explicit constructor arguments take priority.
        # Otherwise use environment variables.
        qdrant_url = (
            url
            if url is not None
            else os.getenv("QDRANT_URL")
        )

        qdrant_api_key = (
            api_key
            if api_key is not None
            else os.getenv("QDRANT_API_KEY")
        )

        # --------------------------------------------------
        # Remote Qdrant / Qdrant Cloud
        # --------------------------------------------------

        if qdrant_url:

            print(
                "[QDRANT] Connecting to remote Qdrant..."
            )

            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )

            print(
                "[QDRANT] Remote Qdrant connection configured."
            )

        # --------------------------------------------------
        # Local persistent Qdrant
        # --------------------------------------------------

        else:

            path = (
                Path(storage_path)
                if storage_path
                else DEFAULT_STORAGE_PATH
            )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                f"[QDRANT] Using local storage: {path}"
            )

            self.client = QdrantClient(
                path=str(path)
            )

        self._ensure_collection()

    # ------------------------------------------------------
    # Collection management
    # ------------------------------------------------------

    def _ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.

        If the collection already exists, verify that its
        vector dimension matches the configured dimension.
        """

        collections = self.client.get_collections()

        collection_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in collection_names:

            print(
                f"[QDRANT] Creating collection: "
                f"{self.collection_name}"
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            print(
                f"[QDRANT] Collection created: "
                f"{self.collection_name}"
            )

            return

        # --------------------------------------------------
        # Validate existing collection configuration
        # --------------------------------------------------

        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )

        vectors_config = (
            collection_info.config.params.vectors
        )

        # Qdrant can represent vectors configuration in
        # different forms depending on the client/version.
        if isinstance(
            vectors_config,
            models.VectorParams,
        ):

            existing_size = vectors_config.size

            if existing_size != self.vector_size:

                raise ValueError(
                    f"Qdrant collection "
                    f"'{self.collection_name}' "
                    f"uses vector dimension "
                    f"{existing_size}, "
                    f"but this application expects "
                    f"{self.vector_size}. "
                    f"The collection must be recreated."
                )

        print(
            f"[QDRANT] Collection ready: "
            f"{self.collection_name}"
        )

    # ------------------------------------------------------
    # Upsert
    # ------------------------------------------------------

    def upsert(
        self,
        embeddings: List[List[float]],
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Insert or update document vectors in Qdrant.
        """

        if len(embeddings) != len(documents):

            raise ValueError(
                "Number of embeddings must match "
                "number of documents."
            )

        if not embeddings:
            return

        points: List[models.PointStruct] = []

        for embedding, document in zip(
            embeddings,
            documents,
        ):

            if len(embedding) != self.vector_size:

                raise ValueError(
                    f"Embedding dimension "
                    f"{len(embedding)} "
                    f"does not match Qdrant "
                    f"vector size "
                    f"{self.vector_size}."
                )

            chunk_id = document.get(
                "chunk_id"
            )

            if not chunk_id:

                raise ValueError(
                    "Document is missing chunk_id."
                )

            metadata = document.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):

                metadata = {}

            payload = {
                "text": document.get(
                    "text",
                    "",
                ),
                **metadata,
            }

            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # ------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Qdrant using cosine similarity.
        """

        if not query_vector:

            raise ValueError(
                "Query vector cannot be empty."
            )

        if len(query_vector) != self.vector_size:

            raise ValueError(
                f"Query vector dimension "
                f"{len(query_vector)} does not match "
                f"Qdrant vector size "
                f"{self.vector_size}."
            )

        if limit <= 0:

            raise ValueError(
                "limit must be greater than zero."
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        formatted_results: List[
            Dict[str, Any]
        ] = []

        for result in results:

            payload = result.payload or {}

            formatted_results.append(
                {
                    "chunk_id": str(
                        result.id
                    ),
                    "score": float(
                        result.score
                    ),
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "metadata": {
                        key: value
                        for key, value
                        in payload.items()
                        if key != "text"
                    },
                }
            )

        return formatted_results

    # ------------------------------------------------------
    # Collection count
    # ------------------------------------------------------

    def count(self) -> int:
        """
        Return the exact number of stored vectors.
        """

        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    # ------------------------------------------------------
    # Delete collection
    # ------------------------------------------------------

    def delete_collection(self) -> None:
        """
        Delete the current Qdrant collection.
        """

        self.client.delete_collection(
            collection_name=self.collection_name
        )

    # ------------------------------------------------------
    # Close
    # ------------------------------------------------------

    def close(self) -> None:
        """
        Close the Qdrant client cleanly.
        """

        if getattr(
            self,
            "client",
            None,
        ) is not None:

            self.client.close()

            self.client = None

    # ------------------------------------------------------
    # Context manager
    # ------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # ------------------------------------------------------
    # Destructor
    # ------------------------------------------------------

    def __del__(self):

        try:

            client = getattr(
                self,
                "client",
                None,
            )

            if client is not None:

                client.close()

        except Exception:

            pass


# ----------------------------------------------------------
# Direct test
# ----------------------------------------------------------

if __name__ == "__main__":

    with QdrantVectorStore() as store:

        print("\n==============================")
        print("MINDO QDRANT VECTOR STORE")
        print("==============================")

        print(
            f"\nCollection: "
            f"{store.collection_name}"
        )

        print(
            f"Vector size: "
            f"{store.vector_size}"
        )

        print(
            f"Distance: COSINE"
        )

        print(
            f"Stored vectors: "
            f"{store.count()}"
        )

        print(
            "\nQdrant vector store initialized successfully."
        )

        print(
            "=============================="
        )