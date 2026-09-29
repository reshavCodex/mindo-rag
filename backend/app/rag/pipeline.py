from typing import Any, Dict, List

from backend.app.rag.ingestion.loader import load_knowledge_base
from backend.app.rag.ingestion.chunker import chunk_documents
from backend.app.rag.ingestion.metadata import enrich_documents
from backend.app.rag.embeddings import GeminiEmbeddingService
from backend.app.rag.vector_store import QdrantVectorStore


class RAGIngestionPipeline:
    """
    MINDO knowledge-base ingestion pipeline.

    Flow:

        Documents
            ↓
        Loader
            ↓
        Cross-page Chunker
            ↓
        Metadata
            ↓
        Gemini Embeddings
            ↓
        Qdrant

    Ingestion strategy:

        Chunks are processed in batches.

        For each batch:

            Embeddings
                ↓
            Qdrant upsert

        This ensures successfully processed batches are stored
        immediately instead of waiting for the entire knowledge
        base to finish embedding.

    Current embedding configuration:

        Model:
            gemini-embedding-2

        Vector dimension:
            1536
    """

    EMBEDDING_MODEL = "gemini-embedding-2"
    VECTOR_SIZE = 1536

    # ------------------------------------------------------
    # Keep this relatively small because Gemini free-tier
    # embedding requests are rate limited.
    # ------------------------------------------------------

    EMBEDDING_BATCH_SIZE = 20

    def __init__(self) -> None:

        self.embedding_service = GeminiEmbeddingService(
            model=self.EMBEDDING_MODEL,
            output_dimension=self.VECTOR_SIZE,
        )

        self.vector_store = QdrantVectorStore(
            collection_name="mindo_knowledge",
            vector_size=self.VECTOR_SIZE,
        )

    def _process_batch(
        self,
        batch_documents: List[Dict[str, Any]],
        batch_number: int,
        total_batches: int,
    ) -> int:
        """
        Generate embeddings for one batch and immediately
        store the resulting vectors in Qdrant.

        Returns:
            Number of vectors successfully stored.
        """

        if not batch_documents:
            return 0

        texts = [
            document["text"]
            for document in batch_documents
        ]

        print(
            f"[Batch {batch_number}/{total_batches}] "
            f"Embedding {len(batch_documents)} chunks..."
        )

        embeddings = (
            self.embedding_service.embed_documents(
                texts
            )
        )

        if len(embeddings) != len(batch_documents):

            raise RuntimeError(
                f"Batch {batch_number}: number of embeddings "
                f"({len(embeddings)}) does not match number "
                f"of documents ({len(batch_documents)})."
            )

        if not embeddings:

            raise RuntimeError(
                f"Batch {batch_number}: Gemini returned "
                f"no embeddings."
            )

        if len(embeddings[0]) != self.VECTOR_SIZE:

            raise RuntimeError(
                f"Batch {batch_number}: unexpected embedding "
                f"dimension {len(embeddings[0])}. "
                f"Expected {self.VECTOR_SIZE}."
            )

        print(
            f"[Batch {batch_number}/{total_batches}] "
            f"Storing {len(embeddings)} vectors in Qdrant..."
        )

        self.vector_store.upsert(
            embeddings=embeddings,
            documents=batch_documents,
        )

        print(
            f"[Batch {batch_number}/{total_batches}] "
            f"SUCCESS"
        )

        return len(embeddings)

    def ingest(self) -> Dict[str, Any]:
        """
        Run the complete knowledge-base ingestion pipeline.

        The ingestion is performed in batches so that vectors
        are stored progressively in Qdrant.

        If Gemini quota is exhausted, already completed batches
        remain stored in Qdrant.
        """

        # --------------------------------------------------
        # Step 1: Load documents
        # --------------------------------------------------

        print(
            "\n[1/5] Loading knowledge base..."
        )

        documents = load_knowledge_base()

        if not documents:

            raise RuntimeError(
                "No documents were loaded "
                "from the knowledge base."
            )

        unique_files = len(
            {
                document.get("file_path")
                for document in documents
                if document.get("file_path")
            }
        )

        print(
            f"Loaded pages/sections: {len(documents)}"
        )

        print(
            f"Unique source files: {unique_files}"
        )

        # --------------------------------------------------
        # Step 2: Chunk documents
        # --------------------------------------------------

        print(
            "\n[2/5] Creating cross-page chunks..."
        )

        chunks = chunk_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200,
        )

        if not chunks:

            raise RuntimeError(
                "No chunks were created "
                "from the loaded documents."
            )

        print(
            f"Created chunks: {len(chunks)}"
        )

        # --------------------------------------------------
        # Step 3: Add metadata
        # --------------------------------------------------

        print(
            "\n[3/5] Adding metadata..."
        )

        enriched_chunks = enrich_documents(
            chunks
        )

        if not enriched_chunks:

            raise RuntimeError(
                "No metadata-enriched chunks "
                "were created."
            )

        print(
            f"Metadata-enriched chunks: "
            f"{len(enriched_chunks)}"
        )

        # --------------------------------------------------
        # Step 4: Generate embeddings in batches
        # --------------------------------------------------

        print(
            "\n[4/5] Generating embeddings "
            "and storing batches..."
        )

        total_chunks = len(
            enriched_chunks
        )

        total_batches = (
            total_chunks
            + self.EMBEDDING_BATCH_SIZE
            - 1
        ) // self.EMBEDDING_BATCH_SIZE

        print(
            f"Embedding batch size: "
            f"{self.EMBEDDING_BATCH_SIZE}"
        )

        print(
            f"Total batches: "
            f"{total_batches}"
        )

        total_embeddings = 0

        for batch_start in range(
            0,
            total_chunks,
            self.EMBEDDING_BATCH_SIZE,
        ):

            batch_end = min(
                batch_start
                + self.EMBEDDING_BATCH_SIZE,
                total_chunks,
            )

            batch_documents = enriched_chunks[
                batch_start:batch_end
            ]

            batch_number = (
                batch_start
                // self.EMBEDDING_BATCH_SIZE
            ) + 1

            stored = self._process_batch(
                batch_documents=batch_documents,
                batch_number=batch_number,
                total_batches=total_batches,
            )

            total_embeddings += stored

            current_qdrant_count = (
                self.vector_store.count()
            )

            print(
                f"Progress: "
                f"{batch_end}/{total_chunks} chunks | "
                f"Qdrant vectors: "
                f"{current_qdrant_count}"
            )

        # --------------------------------------------------
        # Step 5: Final verification
        # --------------------------------------------------

        print(
            "\n[5/5] Verifying Qdrant..."
        )

        stored_count = (
            self.vector_store.count()
        )

        if stored_count < total_chunks:

            raise RuntimeError(
                "Ingestion completed without an exception, "
                "but Qdrant contains fewer vectors than "
                "the number of enriched chunks. "
                f"Expected at least {total_chunks}, "
                f"found {stored_count}."
            )

        print(
            f"Verified vectors stored: "
            f"{stored_count}"
        )

        return {
            "documents": len(documents),
            "unique_files": unique_files,
            "chunks": len(enriched_chunks),
            "embeddings": total_embeddings,
            "vectors_stored": stored_count,
            "embedding_model": self.EMBEDDING_MODEL,
            "vector_dimension": self.VECTOR_SIZE,
            "batch_size": self.EMBEDDING_BATCH_SIZE,
            "batches": total_batches,
        }

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

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()


if __name__ == "__main__":

    with RAGIngestionPipeline() as pipeline:

        result = pipeline.ingest()

        print(
            "\n========================================"
        )

        print(
            "       RAG INGESTION COMPLETE"
        )

        print(
            "========================================"
        )

        print(
            f"Documents / pages : "
            f"{result['documents']}"
        )

        print(
            f"Unique files      : "
            f"{result['unique_files']}"
        )

        print(
            f"Chunks            : "
            f"{result['chunks']}"
        )

        print(
            f"Embeddings        : "
            f"{result['embeddings']}"
        )

        print(
            f"Vectors stored    : "
            f"{result['vectors_stored']}"
        )

        print(
            f"Embedding model   : "
            f"{result['embedding_model']}"
        )

        print(
            f"Vector dimension  : "
            f"{result['vector_dimension']}"
        )

        print(
            f"Batch size        : "
            f"{result['batch_size']}"
        )

        print(
            f"Batches           : "
            f"{result['batches']}"
        )

        print(
            "========================================"
        )

        print(
            "STATUS: SUCCESS"
        )

        print(
            "========================================"
        )