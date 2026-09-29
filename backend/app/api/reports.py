import base64
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.report.pdf_generator import PDFReportGenerator


router = APIRouter(
    prefix="/api/v1/reports",
    tags=["reports"],
)


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

        generator = PDFReportGenerator(
            output_directory="reports"
        )

        output_path, assessment_result = (
            generator.generate_from_context(
                context=context
            )
        )

        # --------------------------------------------------------
        # Read the generated PDF while it still exists inside
        # the RAG service.
        #
        # The Conversation backend runs as a separate service
        # in production, so it cannot access this filesystem.
        # --------------------------------------------------------

        if not output_path.exists():
            raise RuntimeError(
                f"Generated PDF not found: {output_path}"
            )

        report_bytes = output_path.read_bytes()

        if not report_bytes:
            raise RuntimeError(
                "Generated PDF is empty."
            )

        report_base64 = base64.b64encode(
            report_bytes
        ).decode("ascii")

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
            "report_filename": output_path.name,
            "report_path": str(
                output_path.resolve()
            ),
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