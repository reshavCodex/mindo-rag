from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid5, NAMESPACE_URL


# Project root:
# D:\MINDO\rag_part
PROJECT_ROOT = Path(__file__).resolve().parents[4]

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


def get_category(file_path: str | Path) -> str:
    """
    Determine the knowledge-base category from the directory
    containing the document.

    Example:
        knowledge_base/assessment/file.pdf
        -> assessment
    """

    path = Path(file_path)

    try:
        relative_path = (
            path.resolve()
            .relative_to(
                KNOWLEDGE_BASE_DIR.resolve()
            )
        )

        if relative_path.parts:
            return relative_path.parts[0]

    except ValueError:
        pass

    return "general"


def create_chunk_id(
    source: str,
    page_start: int | None,
    page_end: int | None,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic unique ID for a chunk.

    The source document, page range, and chunk index are
    included so that the same chunk receives the same ID
    when ingestion is repeated.
    """

    identifier = (
        f"{source}|"
        f"{page_start if page_start is not None else 'na'}|"
        f"{page_end if page_end is not None else 'na'}|"
        f"{chunk_index}"
    )

    return str(
        uuid5(
            NAMESPACE_URL,
            identifier,
        )
    )


def enrich_chunk_metadata(
    chunk: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add production-ready metadata to a single RAG chunk.

    Supports cross-page chunks through:

        page_start
        page_end

    Also supports the old `page` field for backward
    compatibility.
    """

    source = chunk.get(
        "source",
        "unknown",
    )

    # ------------------------------------------------------
    # New cross-page provenance
    # ------------------------------------------------------

    page_start = chunk.get(
        "page_start"
    )

    page_end = chunk.get(
        "page_end"
    )

    # ------------------------------------------------------
    # Backward compatibility with the old page field.
    # ------------------------------------------------------

    if (
        page_start is None
        and page_end is None
    ):

        old_page = chunk.get(
            "page"
        )

        page_start = old_page
        page_end = old_page

    chunk_index = chunk.get(
        "chunk_index",
        0,
    )

    file_path = chunk.get(
        "file_path"
    )

    chunk_id = create_chunk_id(
        source=source,
        page_start=page_start,
        page_end=page_end,
        chunk_index=chunk_index,
    )

    category = (
        get_category(file_path)
        if file_path
        else "general"
    )

    return {
        "chunk_id": chunk_id,
        "text": chunk.get(
            "text",
            "",
        ),
        "metadata": {
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
            "category": category,
            "file_path": file_path,
        },
    }


def enrich_documents(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add metadata to all RAG chunks.
    """

    enriched_chunks = []

    for chunk in chunks:

        enriched = enrich_chunk_metadata(
            chunk
        )

        if enriched["text"].strip():

            enriched_chunks.append(
                enriched
            )

    return enriched_chunks


if __name__ == "__main__":

    sample_chunk = {
        "text": (
            "Mental health includes emotional, "
            "psychological, and social well-being. "
            "Self-care can support mental health."
        ),
        "source": (
            "caring_for_your_mental_health_nimh.pdf"
        ),
        "page_start": 1,
        "page_end": 2,
        "file_path": str(
            KNOWLEDGE_BASE_DIR
            / "assessment"
            / "caring_for_your_mental_health_nimh.pdf"
        ),
        "chunk_index": 0,
    }

    result = enrich_chunk_metadata(
        sample_chunk
    )

    print("\n==============================")
    print("MINDO METADATA")
    print("==============================")

    print("\nChunk ID:")
    print(result["chunk_id"])

    print("\nText:")
    print(result["text"])

    print("\nMetadata:")

    for key, value in result["metadata"].items():

        print(
            f"{key}: {value}"
        )