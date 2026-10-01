import base64
import os
import re
import tempfile
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.report.pdf_generator import PDFReportGenerator


router = APIRouter(
    prefix="/api/v1/reports",
    tags=["reports"],
)


# ============================================================
# CONCURRENCY CONTROL
#
# Many users may request reports at the same time, and each
# report calls Gemini and Cohere with shared API keys.
#
# This is NOT a global lock: several reports run in parallel.
# It only caps how many run AT ONCE so CPU and API quotas are
# protected. Extra requests wait their turn; if they wait too
# long they receive a clear 503 instead of hanging forever.
#
# Both values can be tuned from Render environment variables.
# ============================================================

def _env_int(
    name: str,
    default: int,
    minimum: int = 1,
) -> int:

    raw = os.getenv(name)

    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return max(minimum, value)


MAX_CONCURRENT_REPORTS = _env_int(
    "MINDO_MAX_CONCURRENT_REPORTS",
    4,
)

QUEUE_TIMEOUT_SECONDS = _env_int(
    "MINDO_REPORT_QUEUE_TIMEOUT_SECONDS",
    300,
)

_report_slots = threading.BoundedSemaphore(
    MAX_CONCURRENT_REPORTS
)


# ============================================================
# FILENAME SAFETY
# ============================================================

_UNSAFE_FILENAME_CHARS = re.compile(
    r"[^A-Za-z0-9_-]"
)


def _safe_filename_part(
    value: Any,
) -> str:
    """
    Reduce a session id to characters that are safe in a
    filename, so values such as "../../x" can never escape
    the temporary directory.

    Normal ids (for example UUIDs) are unchanged.
    """

    cleaned = _UNSAFE_FILENAME_CHARS.sub(
        "_",
        str(value),
    )[:100]

    return cleaned or "report"


@router.post("/generate")
def generate_report(
    context: dict[str, Any],
):
    """
    Generate a MINDO assessment PDF from
    a semantic_context.json payload.

    The generated PDF is returned as base64 data so that
    the Conversation backend can receive the actual PDF
    across the network without depending on the RAG
    service's local filesystem.

    Concurrency:
    - The shared RAG pipeline is built once at startup.
      A request that arrives while it is still warming up
      waits for that same build.
    - Each request writes its PDF into its OWN temporary
      directory, which is deleted as soon as the bytes have
      been read. Requests can never overwrite each other.
    """

    try:
        if not context:
            raise ValueError(
                "Semantic context cannot be empty."
            )

        session = context.get(
            "session",
            {},
        )

        if not isinstance(session, dict):
            raise ValueError(
                "Invalid session data."
            )

        session_id = session.get(
            "session_id"
        )

        if not session_id:
            raise ValueError(
                "semantic_context is missing "
                "session.session_id"
            )

        # --------------------------------------------------------
        # Wait for a free report slot (bounded).
        # --------------------------------------------------------

        queued_at = time.perf_counter()

        acquired = _report_slots.acquire(
            timeout=QUEUE_TIMEOUT_SECONDS
        )

        if not acquired:

            print(
                "[RAG API] Report queue timeout for session:",
                session_id,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "The report service is busy. "
                    "Please retry shortly."
                ),
                headers={
                    "Retry-After": "30",
                },
            )

        started_at = time.perf_counter()

        print(
            f"[RAG API] Report started: "
            f"session={session_id} "
            f"queue_wait={started_at - queued_at:.1f}s"
        )

        try:

            # ----------------------------------------------------
            # Per-request temporary directory.
            #
            # The directory is removed automatically when this
            # block exits, whether the request succeeds or fails,
            # so PDFs never accumulate on disk and two requests
            # (even with the same session id) never share a file.
            # ----------------------------------------------------

            with tempfile.TemporaryDirectory(
                prefix="mindo_report_"
            ) as temp_directory:

                generator = PDFReportGenerator(
                    output_directory=temp_directory
                )

                filename = (
                    "mindo_assessment_"
                    f"{_safe_filename_part(session_id)}"
                    ".pdf"
                )

                output_path, assessment_result = (
                    generator.generate_from_context(
                        context=context,
                        filename=filename,
                    )
                )

                # ------------------------------------------------
                # Read the generated PDF while it still exists.
                #
                # The Conversation backend runs as a separate
                # service in production, so it cannot access
                # this filesystem.
                # ------------------------------------------------

                if not output_path.exists():
                    raise RuntimeError(
                        f"Generated PDF not found: "
                        f"{output_path}"
                    )

                report_bytes = output_path.read_bytes()

                report_filename = output_path.name

                report_path = str(
                    output_path.resolve()
                )

        finally:

            _report_slots.release()

        if not report_bytes:
            raise RuntimeError(
                "Generated PDF is empty."
            )

        report_base64 = base64.b64encode(
            report_bytes
        ).decode("ascii")

        print(
            f"[RAG API] Report finished: "
            f"session={session_id} "
            f"duration="
            f"{time.perf_counter() - started_at:.1f}s"
        )

        # --------------------------------------------------------
        # The AssessmentEngine returns:
        #
        # {
        #     "assessment": {
        #         "category": ...,
        #         "confidence": ...,
        #         "summary": ...
        #     },
        #     ...
        #     "recommendations": [...]
        # }
        #
        # Therefore:
        # - category/confidence/summary are inside "assessment"
        # - recommendations are at the top level
        # --------------------------------------------------------

        if not isinstance(
            assessment_result,
            dict,
        ):
            assessment_result = {}

        assessment = assessment_result.get(
            "assessment",
            {},
        )

        if not isinstance(
            assessment,
            dict,
        ):
            assessment = {}

        recommendations = assessment_result.get(
            "recommendations",
            [],
        )

        if not isinstance(
            recommendations,
            list,
        ):
            recommendations = []

        return {
            "status": "success",
            "session_id": session_id,
            "report_filename": report_filename,
            "report_path": report_path,
            "report_base64": report_base64,
            "assessment": {
                "category": assessment.get(
                    "category"
                ),
                "confidence": assessment.get(
                    "confidence"
                ),
                "summary": assessment.get(
                    "summary"
                ),
                "recommendations": recommendations,
            },
        }

    except HTTPException:

        # Already a deliberate HTTP response (for example the
        # 503 "busy" response). Do not convert it into a 500.
        raise

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        print(
            "[RAG API] Report generation failed:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Report generation failed: "
                f"{error}"
            ),
        ) from error