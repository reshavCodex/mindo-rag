from pathlib import Path
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[4]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_pdf(file_path: Path) -> list[dict]:
    """
    Load a PDF while preserving page-level provenance.

    Each page is returned as a separate source section.
    The chunker can later combine these sections across
    page boundaries while retaining page information.
    """

    documents = []

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            {
                "text": text.strip(),
                "source": file_path.name,
                "page": page_number,
                "file_path": str(file_path),
            }
        )

    return documents


def load_text(file_path: Path) -> list[dict]:
    """
    Load TXT or Markdown files.

    Text-based files do not have page numbers, so page
    provenance is represented as None.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if not text.strip():
        return []

    return [
        {
            "text": text.strip(),
            "source": file_path.name,
            "page": None,
            "file_path": str(file_path),
        }
    ]


def load_document(file_path: Path) -> list[dict]:
    """
    Load a single supported document.
    """

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension in {".txt", ".md"}:
        return load_text(file_path)

    return []


def load_knowledge_base() -> list[dict]:
    """
    Recursively load all supported documents from the
    knowledge base.

    PDF pages remain separate at this stage so their
    provenance is preserved. Cross-page chunking is handled
    by the chunker.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_DIR}"
        )

    documents = []

    for file_path in KNOWLEDGE_BASE_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        try:

            loaded = load_document(
                file_path
            )

            documents.extend(
                loaded
            )

            print(
                f"Loaded: {file_path.name} "
                f"({len(loaded)} section/page(s))"
            )

        except Exception as e:

            print(
                f"Failed to load "
                f"{file_path}: {e}"
            )

    print(
        f"\nTotal loaded sections/pages: "
        f"{len(documents)}"
    )

    return documents


if __name__ == "__main__":

    docs = load_knowledge_base()

    for doc in docs[:3]:

        print("\n---")
        print(
            "Source:",
            doc["source"],
        )
        print(
            "Page:",
            doc["page"],
        )
        print(
            "Text:",
            doc["text"][:300],
        )