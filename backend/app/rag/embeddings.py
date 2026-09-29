from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from time import monotonic, sleep
from typing import List

from google import genai
from google.genai import types

from backend.app.config import settings


class GeminiEmbeddingService:
    """
    Gemini embedding service for MINDO RAG.

    Model:
        gemini-embedding-2

    Vector space:
        1536 dimensions

    The service uses controlled concurrency and rate limiting
    to avoid exhausting Gemini embedding request quotas.
    """

    DEFAULT_MODEL = "gemini-embedding-2"
    DEFAULT_DIMENSION = 1536

    # ------------------------------------------------------
    # Gemini free-tier request limit observed in the project:
    #
    # 100 requests / minute
    #
    # Keep a safety margin instead of targeting the exact
    # provider limit.
    # ------------------------------------------------------

    DEFAULT_MAX_REQUESTS_PER_MINUTE = 80

    DEFAULT_MAX_WORKERS = 2

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        output_dimension: int = DEFAULT_DIMENSION,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_requests_per_minute: int = (
            DEFAULT_MAX_REQUESTS_PER_MINUTE
        ),
    ) -> None:

        self.model = model
        self.output_dimension = output_dimension
        self.max_workers = max_workers
        self.max_requests_per_minute = (
            max_requests_per_minute
        )

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        if self.output_dimension <= 0:
            raise ValueError(
                "output_dimension must be greater than zero."
            )

        if self.max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than zero."
            )

        if self.max_requests_per_minute <= 0:
            raise ValueError(
                "max_requests_per_minute must be greater "
                "than zero."
            )

        if self.max_requests_per_minute > 100:
            raise ValueError(
                "max_requests_per_minute cannot exceed "
                "the currently supported Gemini free-tier "
                "limit of 100 requests per minute."
            )

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=30000
            ),
        )

        # --------------------------------------------------
        # Shared request timestamps.
        #
        # Multiple worker threads use the same limiter.
        # --------------------------------------------------

        self._rate_lock = Lock()

        self._request_times: List[float] = []

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def _validate_text(
        self,
        text: str,
    ) -> None:

        if not text or not text.strip():

            raise ValueError(
                "Text cannot be empty."
            )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------

    def _format_document(
        self,
        text: str,
    ) -> str:

        self._validate_text(text)

        return (
            "title: MINDO Knowledge Base | "
            f"text: {text.strip()}"
        )

    def _format_query(
        self,
        query: str,
    ) -> str:

        self._validate_text(query)

        return (
            "task: search result | "
            f"query: {query.strip()}"
        )

    # ------------------------------------------------------
    # Embedding configuration
    # ------------------------------------------------------

    def _embedding_config(
        self,
    ) -> types.EmbedContentConfig:

        return types.EmbedContentConfig(
            output_dimensionality=self.output_dimension,
        )

    # ------------------------------------------------------
    # Embedding validation
    # ------------------------------------------------------

    def _validate_embedding(
        self,
        embedding: List[float],
        item_name: str,
    ) -> List[float]:

        if not embedding:

            raise RuntimeError(
                f"Gemini returned an empty embedding "
                f"for {item_name}."
            )

        if len(embedding) != self.output_dimension:

            raise RuntimeError(
                f"Unexpected embedding dimension for "
                f"{item_name}: {len(embedding)}. "
                f"Expected: {self.output_dimension}."
            )

        return embedding

    # ------------------------------------------------------
    # Rate limiter
    # ------------------------------------------------------

    def _wait_for_rate_limit(
        self,
    ) -> None:
        """
        Allow only the configured number of requests
        during a rolling 60-second window.

        A shared lock makes this safe when multiple worker
        threads are generating embeddings concurrently.
        """

        while True:

            wait_time = 0.0

            with self._rate_lock:

                now = monotonic()

                cutoff = (
                    now - 60.0
                )

                self._request_times = [
                    timestamp
                    for timestamp
                    in self._request_times
                    if timestamp > cutoff
                ]

                if (
                    len(self._request_times)
                    < self.max_requests_per_minute
                ):

                    self._request_times.append(
                        now
                    )

                    return

                oldest_request = min(
                    self._request_times
                )

                wait_time = max(
                    0.1,
                    60.0
                    - (
                        now
                        - oldest_request
                    ),
                )

            sleep(wait_time)

    # ------------------------------------------------------
    # Single document embedding
    # ------------------------------------------------------

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate one document embedding.

        Every API request passes through the shared
        rate limiter.
        """

        formatted_text = (
            self._format_document(text)
        )

        self._wait_for_rate_limit()

        response = (
            self.client.models.embed_content(
                model=self.model,
                contents=formatted_text,
                config=self._embedding_config(),
            )
        )

        if not response.embeddings:

            raise RuntimeError(
                "Gemini returned no embedding."
            )

        embedding = (
            response.embeddings[0].values
        )

        return self._validate_embedding(
            embedding,
            "document",
        )

    # ------------------------------------------------------
    # Worker
    # ------------------------------------------------------

    def _embed_single_document(
        self,
        index: int,
        text: str,
    ) -> tuple[int, List[float]]:

        embedding = self.embed_text(
            text
        )

        return index, embedding

    # ------------------------------------------------------
    # Multiple document embeddings
    # ------------------------------------------------------

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate multiple document embeddings.

        Requests are executed with controlled concurrency,
        while the shared rate limiter prevents the process
        from exceeding the configured request rate.

        The returned list preserves the original order.
        """

        if not texts:
            return []

        for text in texts:

            self._validate_text(
                text
            )

        embeddings: List[
            List[float] | None
        ] = [
            None
            for _ in texts
        ]

        worker_count = min(
            self.max_workers,
            len(texts),
        )

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            futures = {
                executor.submit(
                    self._embed_single_document,
                    index,
                    text,
                ): index
                for index, text
                in enumerate(texts)
            }

            try:

                completed = 0

                total = len(
                    futures
                )

                for future in as_completed(
                    futures
                ):

                    index, embedding = (
                        future.result()
                    )

                    embeddings[index] = (
                        embedding
                    )

                    completed += 1

                    if (
                        completed == total
                        or completed % 10 == 0
                    ):

                        print(
                            f"Embedding progress: "
                            f"{completed}/{total}"
                        )

            except Exception:

                for future in futures:
                    future.cancel()

                raise

        final_embeddings: List[
            List[float]
        ] = []

        for index, embedding in enumerate(
            embeddings
        ):

            if embedding is None:

                raise RuntimeError(
                    f"Missing embedding for "
                    f"document index {index}."
                )

            final_embeddings.append(
                embedding
            )

        if len(final_embeddings) != len(
            texts
        ):

            raise RuntimeError(
                "Number of generated embeddings "
                "does not match the number of "
                "input documents."
            )

        return final_embeddings

    # ------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------

    def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Generate a retrieval-query embedding.

        Query requests also pass through the same
        rate limiter.
        """

        formatted_query = (
            self._format_query(query)
        )

        self._wait_for_rate_limit()

        response = (
            self.client.models.embed_content(
                model=self.model,
                contents=formatted_query,
                config=self._embedding_config(),
            )
        )

        if not response.embeddings:

            raise RuntimeError(
                "Gemini returned no query embedding."
            )

        embedding = (
            response.embeddings[0].values
        )

        return self._validate_embedding(
            embedding,
            "query",
        )


if __name__ == "__main__":

    import time

    service = GeminiEmbeddingService()

    documents = [
        "Mental health includes emotional, psychological, and social well-being.",
        "Regular physical activity can support mental wellbeing.",
        "Maintaining social connections can provide emotional support.",
        "Sufficient sleep is important for overall wellbeing.",
        "Stress can affect sleep, concentration, and daily functioning.",
    ]

    query = (
        "What are some ways to manage stress "
        "and improve sleep?"
    )

    print("\n==============================")
    print("MINDO GEMINI EMBEDDINGS")
    print("==============================")

    print(
        f"\nModel: {service.model}"
    )

    print(
        f"Output dimension: "
        f"{service.output_dimension}"
    )

    print(
        f"Max concurrent workers: "
        f"{service.max_workers}"
    )

    print(
        f"Max requests/minute: "
        f"{service.max_requests_per_minute}"
    )

    print(
        "\nTesting query embedding..."
    )

    start = time.perf_counter()

    query_vector = (
        service.embed_query(
            query
        )
    )

    query_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Query embedding dimension: "
        f"{len(query_vector)}"
    )

    print(
        f"Query embedding time: "
        f"{query_time:.2f} sec"
    )

    print(
        "\nTesting single document embedding..."
    )

    start = time.perf_counter()

    vector = service.embed_text(
        documents[0]
    )

    single_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Single document dimension: "
        f"{len(vector)}"
    )

    print(
        f"Single document embedding time: "
        f"{single_time:.2f} sec"
    )

    print(
        "\nTesting controlled document embedding..."
    )

    start = time.perf_counter()

    embeddings = (
        service.embed_documents(
            documents
        )
    )

    controlled_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Embeddings returned: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    print(
        f"Controlled embedding time: "
        f"{controlled_time:.2f} sec"
    )

    print("\n==============================")
    print("EMBEDDING TEST COMPLETE")
    print("==============================")