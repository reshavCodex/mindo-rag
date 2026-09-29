from __future__ import annotations

import json
import time
from functools import lru_cache
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

    The object is designed to be created once and reused
    across multiple assessment requests.
    """

    def __init__(self) -> None:
        print("Initializing MINDO RAG pipeline...")

        self.query_builder = RAGQueryBuilder()
        self.retrieval_pipeline = RAGRetrievalPipeline()
        self.final_llm = FinalLLM()
        self.assessment_engine = AssessmentEngine()

        print("MINDO RAG pipeline initialized.")

    def run(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run the complete MINDO assessment pipeline.
        """

        if not context:
            raise ValueError(
                "Context cannot be empty."
            )

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

    def close(self) -> None:
        """
        Release resources held by the retrieval pipeline.
        """

        self.retrieval_pipeline.close()


@lru_cache(maxsize=1)
def get_mindo_rag() -> MINDORAG:
    """
    Return the shared MINDO RAG instance.

    The singleton prevents expensive components such as the
    BM25 index and Qdrant connection from being rebuilt for
    every request.
    """

    return MINDORAG()


def close_mindo_rag() -> None:
    """
    Close the shared MINDO RAG instance and clear its cache.
    """

    if get_mindo_rag.cache_info().currsize == 0:
        return

    rag = get_mindo_rag()

    try:
        rag.close()
    finally:
        get_mindo_rag.cache_clear()


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