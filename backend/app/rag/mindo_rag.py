from __future__ import annotations

import json
import threading
import time
from typing import Any

from backend.app.rag.query_builder import RAGQueryBuilder
from backend.app.rag.retrieval_pipeline import RAGRetrievalPipeline
from backend.app.ai.final_llm import FinalLLM
from backend.app.assessment.engine import AssessmentEngine


class MINDORAG:
    """
    Complete MINDO assessment pipeline.

    Flow:

        Context Builder
              ↓
        Query Builder
              ↓
        Hybrid Retrieval
              ↓
        Cohere Reranker
              ↓
        Final LLM
              ↓
        Assessment Engine

    The object is designed to be created ONCE per process and
    shared by every concurrent assessment request.

    Concurrency model
    -----------------
    - run() keeps ALL request data (context, query, evidence,
      analysis, assessment) in local variables. Nothing about a
      request is ever stored on the shared object, so concurrent
      calls cannot see each other's data.
    - The components built in __init__ (BM25 index, Qdrant client,
      Gemini / Cohere clients) are only READ while serving requests.
    - close() is idempotent and safe to call while requests are
      running: it stops accepting new runs, waits for in-flight runs
      to finish, and only then releases resources.
    """

    # Maximum time close() waits for in-flight runs to finish
    # before releasing resources anyway.
    CLOSE_DRAIN_TIMEOUT_SECONDS = 60.0

    def __init__(self) -> None:
        print("Initializing MINDO RAG pipeline...")

        self.query_builder = RAGQueryBuilder()
        self.retrieval_pipeline = RAGRetrievalPipeline()
        self.final_llm = FinalLLM()
        self.assessment_engine = AssessmentEngine()

        # Lifecycle state only (never request data).
        self._state = threading.Condition()
        self._active_runs = 0
        self._closed = False

        print("MINDO RAG pipeline initialized.")

    def run(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the complete MINDO assessment pipeline.

        Safe to call concurrently from multiple threads.
        """

        if not context:
            raise ValueError(
                "Context cannot be empty."
            )

        # Register this run so close() cannot release resources
        # underneath it.
        with self._state:
            if self._closed:
                raise RuntimeError(
                    "MINDO RAG pipeline is closed."
                )

            self._active_runs += 1

        try:

            # --------------------------------------------
            # 1. Build retrieval query
            # --------------------------------------------

            query = self.query_builder.build_query(
                context
            )

            # --------------------------------------------
            # 2. Retrieve and rerank evidence
            # --------------------------------------------

            evidence = self.retrieval_pipeline.retrieve(
                query=query
            )

            # --------------------------------------------
            # 3. Final LLM evidence synthesis
            # --------------------------------------------

            analysis = self.final_llm.analyze(
                context=context,
                evidence=evidence,
            )

            # --------------------------------------------
            # 4. Assessment Engine
            # --------------------------------------------

            assessment = self.assessment_engine.assess(
                analysis=analysis,
                context=context,
            )

            # --------------------------------------------
            # 5. Return complete pipeline result
            # --------------------------------------------

            return {
                "query": query,
                "evidence": evidence,
                "analysis": analysis,
                "assessment": assessment,
            }

        finally:

            with self._state:
                self._active_runs -= 1
                self._state.notify_all()

    def close(self) -> None:
        """
        Release resources held by the retrieval pipeline.

        Idempotent: calling close() more than once is harmless.
        New runs are rejected immediately; in-flight runs are
        allowed to finish (up to CLOSE_DRAIN_TIMEOUT_SECONDS)
        before resources are released.
        """

        with self._state:

            if self._closed:
                return

            self._closed = True

            drained = self._state.wait_for(
                lambda: self._active_runs == 0,
                timeout=self.CLOSE_DRAIN_TIMEOUT_SECONDS,
            )

            if not drained:
                print(
                    "MINDO RAG close(): timed out waiting for "
                    f"{self._active_runs} in-flight run(s); "
                    "releasing resources anyway."
                )

        self.retrieval_pipeline.close()


# ------------------------------------------------------------
# Shared (process-wide) instance
#
# Built exactly once, even if many threads ask for it at the
# same time. Threads that arrive while the first build is in
# progress wait on the lock and then receive the same instance.
# ------------------------------------------------------------

_shared_rag: MINDORAG | None = None
_shared_rag_lock = threading.Lock()


def get_mindo_rag() -> MINDORAG:
    """
    Return the shared MINDO RAG instance, building it on first use.

    The singleton prevents expensive components such as the
    knowledge-base load, the BM25 index and the Qdrant connection
    from being rebuilt for every request.

    Thread-safe: concurrent first calls build the pipeline once.
    If the build fails, nothing is cached and the next call retries.
    """

    global _shared_rag

    # Fast path (no lock) once the pipeline is built.
    rag = _shared_rag

    if rag is not None:
        return rag

    with _shared_rag_lock:

        # Another thread may have finished building while this
        # thread was waiting for the lock.
        if _shared_rag is None:
            _shared_rag = MINDORAG()

        return _shared_rag


def is_mindo_rag_ready() -> bool:
    """
    Return True once the shared MINDO RAG instance has been built.

    Never triggers a build, so it is safe for health/readiness checks.
    """

    return _shared_rag is not None


def close_mindo_rag() -> None:
    """
    Close the shared MINDO RAG instance and clear it.

    Intended to be called ONCE at application shutdown.
    Idempotent: does nothing if no instance exists.
    """

    global _shared_rag

    with _shared_rag_lock:
        rag = _shared_rag
        _shared_rag = None

    if rag is None:
        return

    # Outside the lock: close() may wait for in-flight runs and
    # must not block other threads from calling get_mindo_rag().
    rag.close()


def run_mindo_rag(
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function for running the shared MINDO RAG instance.
    """

    rag = get_mindo_rag()

    return rag.run(context)


def _build_test_context() -> dict[str, Any]:
    """
    Build a small representative Context Builder output
    for local pipeline testing.
    """

    return {
        "schema_version": "2.0",

        "session": {
            "session_id": "mindo-test-session",
            "duration_seconds": 90,
        },

        "conversation": {
            "turns": [
                {
                    "turn_number": 1,
                    "user_text": (
                        "I have been feeling stressed because "
                        "my exams are coming up."
                    ),
                    "assistant_text": (
                        "Can you tell me how this stress has "
                        "been affecting you?"
                    ),
                },
                {
                    "turn_number": 2,
                    "user_text": (
                        "I haven't been sleeping properly "
                        "and I keep worrying about the exam."
                    ),
                    "assistant_text": (
                        "That sounds difficult. How has this "
                        "been affecting your daily routine?"
                    ),
                },
                {
                    "turn_number": 3,
                    "user_text": (
                        "I find it difficult to concentrate "
                        "when I study."
                    ),
                    "assistant_text": (
                        "Thank you for sharing that."
                    ),
                },
            ]
        },

        "behavioral_signals": {
            "turn_observations": [
                {
                    "turn_number": 1,
                    "dominant_expression": "neutral",
                    "probabilities": {
                        "neutral": 0.70,
                        "sad": 0.10,
                        "angry": 0.05,
                        "happy": 0.15,
                    },
                    "observation_count": 5,
                },
                {
                    "turn_number": 2,
                    "dominant_expression": "neutral",
                    "probabilities": {
                        "neutral": 0.65,
                        "sad": 0.20,
                        "angry": 0.05,
                        "happy": 0.10,
                    },
                    "observation_count": 5,
                },
            ]
        },

        "context": {
            "stated_concerns": [
                "exam stress",
                "sleep difficulty",
                "worry",
                "difficulty concentrating",
            ],
            "situational_factors": [
                "upcoming examinations",
                "academic pressure",
            ],
            "observed_behavioral_patterns": [
                {
                    "signal": "predominantly neutral facial expression",
                    "source": "facial_expression_model",
                    "turns": [1, 2],
                    "confidence": "model_output",
                }
            ],
        },

        "safety": {
            "assessment_status": "not_assessed",
        },
    }


def _run_test() -> None:
    print("\n==============================")
    print("MINDO COMPLETE PIPELINE TEST")
    print("==============================")

    start_total = time.perf_counter()

    rag = get_mindo_rag()

    initialization_time = (
        time.perf_counter() - start_total
    )

    print(
        f"\nInitialization time: "
        f"{initialization_time:.2f} seconds"
    )

    context = _build_test_context()

    print("\nRunning complete pipeline...")

    start_pipeline = time.perf_counter()

    result = rag.run(context)

    pipeline_time = (
        time.perf_counter() - start_pipeline
    )

    total_time = (
        time.perf_counter() - start_total
    )

    print(
        f"Pipeline execution time: "
        f"{pipeline_time:.2f} seconds"
    )

    print(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    print("\n==============================")
    print("PIPELINE RESULT")
    print("==============================")

    print("\n--- QUERY ---")
    print(result["query"])

    print("\n--- EVIDENCE ---")
    print(
        f"Evidence returned: "
        f"{len(result['evidence'])}"
    )

    for index, item in enumerate(
        result["evidence"],
        start=1,
    ):
        print(
            f"\nEvidence {index}:"
        )
        print(
            f"Score: "
            f"{item.get('score')}"
        )
        print(
            f"Source: "
            f"{item.get('source')}"
        )
        print(
            f"Page: "
            f"{item.get('page_start')} - "
            f"{item.get('page_end')}"
        )
        print(
            f"Text: "
            f"{item.get('text', '')[:250]}"
        )

    print("\n--- FINAL LLM ANALYSIS ---")

    print(
        json.dumps(
            result["analysis"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n--- ASSESSMENT ---")

    print(
        json.dumps(
            result["assessment"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n==============================")
    print("MINDO PIPELINE TEST COMPLETE")
    print("==============================")


if __name__ == "__main__":
    try:
        _run_test()
    finally:
        close_mindo_rag()