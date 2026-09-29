from typing import Dict, List
import re


def split_into_sentences(text: str) -> List[str]:
    """
    Split normalized text into sentences.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Create sentence-aware overlapping chunks.

    This function operates on complete document text.
    It does not know about PDF pages.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    sentences = split_into_sentences(
        text
    )

    if not sentences:
        return []

    chunks: List[str] = []

    current_sentences: List[str] = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        additional_length = (
            sentence_length
            if not current_sentences
            else sentence_length + 1
        )

        # --------------------------------------------------
        # If adding the sentence exceeds the chunk size,
        # finalize the current chunk.
        # --------------------------------------------------

        if (
            current_sentences
            and current_length + additional_length
            > chunk_size
        ):

            chunk = " ".join(
                current_sentences
            ).strip()

            if chunk:
                chunks.append(
                    chunk
                )

            # --------------------------------------------------
            # Preserve sentence-level overlap.
            # --------------------------------------------------

            overlap_sentences: List[str] = []
            overlap_length = 0

            for previous_sentence in reversed(
                current_sentences
            ):

                sentence_with_space = (
                    len(previous_sentence) + 1
                )

                if (
                    overlap_sentences
                    and overlap_length
                    + sentence_with_space
                    > chunk_overlap
                ):
                    break

                if (
                    not overlap_sentences
                    and len(previous_sentence)
                    > chunk_overlap
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence,
                )

                overlap_length += (
                    sentence_with_space
                )

            current_sentences = (
                overlap_sentences
            )

            current_length = (
                sum(
                    len(sentence) + 1
                    for sentence
                    in current_sentences
                )
            )

        # --------------------------------------------------
        # Add the current sentence.
        # --------------------------------------------------

        current_sentences.append(
            sentence
        )

        current_length += (
            sentence_length + 1
        )

    # --------------------------------------------------
    # Add final chunk.
    # --------------------------------------------------

    if current_sentences:

        chunk = " ".join(
            current_sentences
        ).strip()

        if chunk:
            chunks.append(
                chunk
            )

    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """
    Create RAG chunks while allowing chunks to cross
    PDF page boundaries.

    Input:
        Page-level documents produced by loader.py.

    Output:
        Document chunks with preserved provenance.

    A chunk may contain text from multiple pages.

    Metadata:
        source
        file_path
        page_start
        page_end
        chunk_index
    """

    if not documents:
        return []

    chunked_documents: List[Dict] = []

    # ------------------------------------------------------
    # Group loaded pages by source document.
    #
    # This prevents chunks from accidentally crossing from
    # one independent document into another.
    # ------------------------------------------------------

    grouped_documents: Dict[str, List[Dict]] = {}

    for document in documents:

        source = document.get(
            "source"
        )

        file_path = document.get(
            "file_path"
        )

        group_key = (
            file_path
            or source
            or "unknown_document"
        )

        grouped_documents.setdefault(
            group_key,
            [],
        ).append(
            document
        )

    # ------------------------------------------------------
    # Process each original document separately.
    # ------------------------------------------------------

    for document_group in grouped_documents.values():

        # Keep the original page/document order.
        document_group.sort(
            key=lambda document: (
                document.get("page")
                if isinstance(
                    document.get("page"),
                    int,
                )
                else 0
            )
        )

        # --------------------------------------------------
        # Combine page text into one document text.
        #
        # This is the key change that allows chunks to cross
        # page boundaries.
        # --------------------------------------------------

        combined_parts: List[str] = []

        for document in document_group:

            text = document.get(
                "text",
                "",
            )

            if (
                isinstance(text, str)
                and text.strip()
            ):
                combined_parts.append(
                    text.strip()
                )

        if not combined_parts:
            continue

        combined_text = "\n\n".join(
            combined_parts
        )

        chunks = chunk_text(
            text=combined_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # --------------------------------------------------
        # Determine page boundaries for every generated
        # chunk.
        #
        # We locate the chunk text inside the combined
        # document text and use character positions to
        # estimate the source pages.
        # --------------------------------------------------

        page_ranges = []

        current_position = 0

        for document in document_group:

            text = document.get(
                "text",
                "",
            )

            if not isinstance(text, str):
                text = ""

            text = text.strip()

            if not text:
                continue

            start = current_position

            end = (
                start
                + len(text)
            )

            page_ranges.append(
                {
                    "start": start,
                    "end": end,
                    "page": document.get(
                        "page"
                    ),
                }
            )

            current_position = (
                end + 2
            )

        # --------------------------------------------------
        # Track approximate position of each generated chunk.
        # --------------------------------------------------

        search_position = 0

        for chunk_index, chunk in enumerate(
            chunks
        ):

            if not chunk:
                continue

            chunk_position = combined_text.find(
                chunk,
                search_position,
            )

            if chunk_position == -1:

                chunk_position = search_position

            chunk_end = (
                chunk_position
                + len(chunk)
            )

            pages = []

            for page_range in page_ranges:

                if (
                    chunk_position
                    < page_range["end"]
                    and chunk_end
                    > page_range["start"]
                ):
                    pages.append(
                        page_range["page"]
                    )

            pages = [
                page
                for page in pages
                if page is not None
            ]

            if pages:

                page_start = min(
                    pages
                )

                page_end = max(
                    pages
                )

            else:

                page_start = None
                page_end = None

            first_document = (
                document_group[0]
            )

            chunked_documents.append(
                {
                    "text": chunk,
                    "source": first_document.get(
                        "source"
                    ),
                    "page_start": page_start,
                    "page_end": page_end,
                    "file_path": first_document.get(
                        "file_path"
                    ),
                    "chunk_index": chunk_index,
                }
            )

            search_position = max(
                chunk_end,
                search_position + 1,
            )

    return chunked_documents


if __name__ == "__main__":

    sample_documents = [
        {
            "text": (
                "Mental health includes emotional, "
                "psychological, and social well-being. "
                "It is more than the absence of mental illness."
            ),
            "source": "example.pdf",
            "page": 1,
            "file_path": "example.pdf",
        },
        {
            "text": (
                "Self-care can play a role in maintaining "
                "mental health. Regular exercise, healthy "
                "meals, hydration, sufficient sleep, and "
                "social connection can support well-being."
            ),
            "source": "example.pdf",
            "page": 2,
            "file_path": "example.pdf",
        },
    ]

    chunks = chunk_documents(
        documents=sample_documents,
        chunk_size=180,
        chunk_overlap=40,
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    for index, chunk in enumerate(
        chunks
    ):

        print(
            f"\n--- Chunk {index} ---"
        )

        print(
            "Page:",
            chunk["page_start"],
            "→",
            chunk["page_end"],
        )

        print(
            "Text:",
            chunk["text"],
        )