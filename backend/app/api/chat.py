from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.rag.mindo_rag import get_mindo_rag
from backend.app.rag.retrieval_pipeline import (
    RAGRetrievalPipeline,
)


# ============================================================
# CHAT ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class ChatRetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User's chat question.",
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ChatRetrieveResponse(BaseModel):
    query: str
    evidence: list[dict[str, Any]]


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================

def get_retrieval_pipeline() -> RAGRetrievalPipeline:
    """
    Return the retrieval pipeline owned by the shared
    MINDO RAG instance.

    The shared instance is built ONCE when the service starts
    (see main.py lifespan) and is used by both the report
    endpoint and this chat endpoint. Chat therefore no longer
    builds its own second copy of the knowledge base, BM25
    index, Qdrant client and Cohere client.

    The shared pipeline uses the same retrieval settings this
    module previously created (candidate_k=20, final_k=5).

    This module must NEVER close the pipeline: its lifecycle
    belongs to the application, and other requests are using
    it at the same time.

    If the shared pipeline is still warming up, the call waits
    for that same build; it never starts a second one.
    """

    return get_mindo_rag().retrieval_pipeline


# ============================================================
# CHAT RETRIEVAL
# ============================================================

@router.post(
    "/retrieve",
    response_model=ChatRetrieveResponse,
)
def retrieve_chat_knowledge(
    request: ChatRetrieveRequest,
):
    """
    Retrieve knowledge-base evidence for MINDO chat.

    IMPORTANT:

    This endpoint ONLY performs retrieval.

    It does NOT:
    - generate an assessment
    - generate a PDF
    - call the assessment engine
    - replace the existing report pipeline

    It reuses the existing MINDO RAG retrieval pipeline.
    """

    try:

        query = request.query.strip()

        if not query:

            raise ValueError(
                "Chat query cannot be empty."
            )

        print()
        print("=" * 70)
        print("MINDO CHAT RETRIEVAL")
        print("=" * 70)

        print(
            f"[QUERY] {query}"
        )

        pipeline = get_retrieval_pipeline()

        evidence = pipeline.retrieve(
            query=query
        )

        print(
            f"[RAG] Evidence returned: "
            f"{len(evidence)}"
        )

        print("=" * 70)

        return {
            "query": query,
            "evidence": evidence,
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        print(
            "[RAG CHAT ERROR]",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Chat retrieval failed: "
                f"{error}"
            ),
        ) from error