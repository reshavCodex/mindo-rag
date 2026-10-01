from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from backend.app.rag.mindo_rag import (
    close_mindo_rag,
    get_mindo_rag,
)


class PDFReportGenerator:
    """
    Generate a structured MINDO assessment PDF report.

    The report is based on the structured output produced
    by the Assessment Engine.

    The report is non-diagnostic and must not present the
    AI output as a clinical diagnosis.
    """

    REPORT_VERSION = "1.0"

    def __init__(
        self,
        output_directory: str | Path = "reports",
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.styles = self._build_styles()

    def _build_styles(
        self,
    ) -> dict[str, ParagraphStyle]:

        base_styles = getSampleStyleSheet()

        return {
            "title": ParagraphStyle(
                "MINDOTitle",
                parent=base_styles["Title"],
                fontSize=22,
                leading=26,
                alignment=TA_CENTER,
                spaceAfter=8 * mm,
            ),
            "subtitle": ParagraphStyle(
                "MINDOSubtitle",
                parent=base_styles["Normal"],
                fontSize=10,
                leading=14,
                alignment=TA_CENTER,
                textColor=colors.grey,
                spaceAfter=10 * mm,
            ),
            "heading": ParagraphStyle(
                "MINDOHeading",
                parent=base_styles["Heading2"],
                fontSize=14,
                leading=18,
                spaceBefore=5 * mm,
                spaceAfter=3 * mm,
            ),
            "body": ParagraphStyle(
                "MINDOBody",
                parent=base_styles["BodyText"],
                fontSize=10,
                leading=15,
                spaceAfter=3 * mm,
            ),
            "small": ParagraphStyle(
                "MINDOSmall",
                parent=base_styles["BodyText"],
                fontSize=8,
                leading=11,
                textColor=colors.grey,
            ),
            "disclaimer": ParagraphStyle(
                "MINDODisclaimer",
                parent=base_styles["BodyText"],
                fontSize=9,
                leading=13,
                spaceBefore=4 * mm,
                spaceAfter=4 * mm,
            ),
        }

    @staticmethod
    def _safe_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return str(value).strip()

    def _paragraph(
        self,
        text: Any,
        style: str = "body",
    ) -> Paragraph:

        safe_text = self._safe_text(text)

        if not safe_text:
            safe_text = "Not provided."

        escaped_text = (
            safe_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        return Paragraph(
            escaped_text,
            self.styles[style],
        )

    def _list_section(
        self,
        title: str,
        items: list[Any],
    ) -> list[Any]:

        elements: list[Any] = []

        elements.append(
            Paragraph(
                title,
                self.styles["heading"],
            )
        )

        if not items:

            elements.append(
                self._paragraph(
                    "None explicitly reported."
                )
            )

            return elements

        for item in items:

            item_text = self._safe_text(item)

            if not item_text:
                continue

            elements.append(
                Paragraph(
                    f"- {item_text}",
                    self.styles["body"],
                )
            )

        return elements

    def _build_session_table(
        self,
        assessment_result: dict[str, Any],
    ) -> Table:

        session = assessment_result.get(
            "session",
            {},
        )

        if not isinstance(
            session,
            dict,
        ):
            session = {}

        assessment_data = assessment_result.get(
            "assessment",
            {},
        )

        if not isinstance(
            assessment_data,
            dict,
        ):
            assessment_data = {}

        duration = session.get(
            "duration_seconds"
        )

        if duration is not None:

            try:

                duration_text = (
                    f"{float(duration):.1f} seconds"
                )

            except (
                TypeError,
                ValueError,
            ):

                duration_text = (
                    f"{duration} seconds"
                )

        else:

            duration_text = "Not provided"

        rows = [
            [
                "Report Version",
                self.REPORT_VERSION,
            ],
            [
                "Session ID",
                session.get(
                    "session_id",
                    "Not provided",
                ),
            ],
            [
                "Duration",
                duration_text,
            ],
            [
                "Assessment Category",
                assessment_data.get(
                    "category",
                    "Not provided",
                ),
            ],
            [
                "Confidence",
                assessment_data.get(
                    "confidence",
                    "Not provided",
                ),
            ],
        ]

        table = Table(
            rows,
            colWidths=[
                55 * mm,
                115 * mm,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (1, 0),
                        (1, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        9,
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ]
            )
        )

        return table

    def _build_story(
        self,
        assessment_result: dict[str, Any],
    ) -> list[Any]:

        story: list[Any] = []

        assessment = assessment_result.get(
            "assessment",
            {},
        )

        observations = assessment_result.get(
            "observations",
            {},
        )

        evidence = assessment_result.get(
            "evidence",
            {},
        )

        safety = assessment_result.get(
            "safety",
            {},
        )

        session = assessment_result.get(
            "session",
            {},
        )

        if not isinstance(
            assessment,
            dict,
        ):
            assessment = {}

        if not isinstance(
            observations,
            dict,
        ):
            observations = {}

        if not isinstance(
            evidence,
            dict,
        ):
            evidence = {}

        if not isinstance(
            safety,
            dict,
        ):
            safety = {}

        if not isinstance(
            session,
            dict,
        ):
            session = {}

        # --------------------------------------------
        # Header
        # --------------------------------------------

        story.append(
            Paragraph(
                "MINDO",
                self.styles["title"],
            )
        )

        story.append(
            Paragraph(
                "AI Mental-Wellness Assessment Report",
                self.styles["subtitle"],
            )
        )

        story.append(
            self._build_session_table(
                assessment_result
            )
        )

        story.append(
            Spacer(
                1,
                6 * mm,
            )
        )

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        story.append(
            Paragraph(
                "Summary",
                self.styles["heading"],
            )
        )

        story.append(
            self._paragraph(
                assessment.get(
                    "summary",
                    "No summary available.",
                )
            )
        )

        # --------------------------------------------
        # Key Observations
        # --------------------------------------------

        story.extend(
            self._list_section(
                "Key Observations",
                observations.get(
                    "key_observations",
                    [],
                ),
            )
        )

        # --------------------------------------------
        # Risk Indicators
        # --------------------------------------------

        story.extend(
            self._list_section(
                "Risk Indicators",
                observations.get(
                    "risk_indicators",
                    [],
                ),
            )
        )

        # --------------------------------------------
        # Protective Factors
        # --------------------------------------------

        story.extend(
            self._list_section(
                "Protective Factors",
                observations.get(
                    "protective_factors",
                    [],
                ),
            )
        )

        # --------------------------------------------
        # Supporting Evidence
        # --------------------------------------------

        story.extend(
            self._list_section(
                "Supporting Evidence",
                evidence.get(
                    "supporting_evidence",
                    [],
                ),
            )
        )

        # --------------------------------------------
        # Recommended Next Steps
        # --------------------------------------------

        story.extend(
            self._list_section(
                "Recommended Next Steps",
                assessment_result.get(
                    "recommendations",
                    [],
                ),
            )
        )

        # --------------------------------------------
        # Safety
        # --------------------------------------------

        story.append(
            Paragraph(
                "Safety",
                self.styles["heading"],
            )
        )

        story.append(
            self._paragraph(
                safety.get(
                    "status",
                    "not_assessed",
                )
            )
        )

        # --------------------------------------------
        # Limitations
        # --------------------------------------------

        story.extend(
            self._list_section(
                "Assessment Limitations",
                assessment_result.get(
                    "limitations",
                    [],
                ),
            )
        )

        # --------------------------------------------
        # Disclaimer
        # --------------------------------------------

        story.append(
            Paragraph(
                "Important Disclaimer",
                self.styles["heading"],
            )
        )

        disclaimer = assessment_result.get(
            "disclaimer",
            (
                "This assessment is an AI-generated, "
                "non-diagnostic summary based on the "
                "information available during the session. "
                "It is not a substitute for evaluation by "
                "a qualified healthcare professional."
            ),
        )

        story.append(
            self._paragraph(
                disclaimer,
                style="disclaimer",
            )
        )

        story.append(
            Spacer(
                1,
                5 * mm,
            )
        )

        story.append(
            Paragraph(
                "Generated by MINDO",
                self.styles["small"],
            )
        )

        return story

    def generate(
        self,
        assessment_result: dict[str, Any],
        filename: str = "mindo_assessment_report.pdf",
    ) -> Path:
        """
        Generate a PDF from an Assessment Engine result.
        """

        if not assessment_result:
            raise ValueError(
                "Assessment result cannot be empty."
            )

        output_path = (
            self.output_directory / filename
        )

        document = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="MINDO Assessment Report",
            author="MINDO",
        )

        story = self._build_story(
            assessment_result
        )

        document.build(story)

        if not output_path.exists():
            raise RuntimeError(
                "PDF generation failed."
            )

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                "Generated PDF is empty."
            )

        return output_path

    def generate_from_context(
        self,
        context: dict[str, Any],
        filename: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """
        Run the real MINDO pipeline from Context Builder output
        and generate a PDF from the Assessment Engine result.

        Flow:

            Context Builder v2.0
                    ↓
                MINDORAG
                    ↓
             Assessment Engine
                    ↓
              PDF Generator
        """

        if not context:
            raise ValueError(
                "Context cannot be empty."
            )

        # --------------------------------------------
        # Run MINDO RAG
        #
        # The RAG pipeline is built ONCE per process
        # (at application startup) and shared by every
        # request. It must NEVER be closed here: other
        # users' requests may be using it at the same time.
        # Its lifecycle is owned by the application
        # (see main.py lifespan).
        # --------------------------------------------

        rag = get_mindo_rag()

        pipeline_result = rag.run(
            context
        )

        if not pipeline_result:
            raise RuntimeError(
                "MINDO RAG returned an empty result."
            )

        # --------------------------------------------
        # IMPORTANT:
        #
        # MINDORAG returns:
        #
        # {
        #     "query": ...,
        #     "evidence": ...,
        #     "analysis": ...,
        #     "assessment": {...}
        # }
        #
        # The PDF must receive the nested
        # Assessment Engine result.
        # --------------------------------------------

        assessment_result = pipeline_result.get(
            "assessment"
        )

        if not isinstance(
            assessment_result,
            dict,
        ):
            raise RuntimeError(
                "MINDO RAG returned an invalid "
                "Assessment Engine result."
            )

        # --------------------------------------------
        # Add Context Builder session information
        # --------------------------------------------

        session = context.get(
            "session",
            {},
        )

        if isinstance(
            session,
            dict,
        ):

            assessment_result = {
                **assessment_result,
                "session": session,
            }

        # --------------------------------------------
        # Generate filename
        # --------------------------------------------

        if not filename:

            session_id = None

            if isinstance(
                session,
                dict,
            ):

                session_id = session.get(
                    "session_id"
                )

            if session_id:

                filename = (
                    "mindo_assessment_"
                    f"{session_id}.pdf"
                )

            else:

                filename = (
                    "mindo_assessment_report.pdf"
                )

        # --------------------------------------------
        # Generate PDF
        # --------------------------------------------

        output_path = self.generate(
            assessment_result=assessment_result,
            filename=filename,
        )

        return (
            output_path,
            assessment_result,
        )


def _build_test_assessment() -> dict[str, Any]:
    """
    Representative Assessment Engine output used only
    for testing PDF generation.
    """

    return {
        "assessment_version": "1.0",

        "session": {
            "session_id": "mindo-pdf-test",
            "duration_seconds": 90,
        },

        "assessment": {
            "category": "wellbeing_concern",
            "confidence": "moderate",
            "summary": (
                "The user reported exam-related stress, "
                "sleep difficulties, persistent worry, "
                "and difficulty concentrating while studying."
            ),
        },

        "observations": {
            "key_observations": [
                (
                    "User explicitly reported stress related "
                    "to upcoming examinations."
                ),
                (
                    "User reported sleep difficulty, worry, "
                    "and difficulty concentrating."
                ),
                (
                    "Behavioral model observed a predominantly "
                    "neutral facial expression."
                ),
            ],
            "risk_indicators": [
                (
                    "Reported sleep difficulties and "
                    "concentration issues associated with "
                    "exam stress."
                ),
            ],
            "protective_factors": [
                (
                    "No protective factors were explicitly "
                    "reported during the session."
                ),
            ],
        },

        "evidence": {
            "supporting_evidence": [
                (
                    "NIMH guidance notes that stress can affect "
                    "sleep, concentration, and daily functioning."
                ),
                (
                    "NIMH guidance highlights the importance "
                    "of sufficient sleep for wellbeing."
                ),
            ],
        },

        "recommendations": [
            (
                "Consider basic stress-management strategies "
                "and healthy sleep practices."
            ),
            (
                "Consider speaking with a qualified healthcare "
                "professional or counselor if difficulties persist."
            ),
        ],

        "safety": {
            "status": "not_assessed",
        },

        "limitations": [
            (
                "The session was brief and does not provide "
                "a complete psychological history."
            ),
            (
                "Facial-expression model outputs are automated "
                "observations and do not prove internal emotional states."
            ),
        ],

        "disclaimer": (
            "This assessment is an AI-generated, non-diagnostic "
            "summary based on the information available during "
            "the session. It is not a substitute for evaluation "
            "by a qualified healthcare professional."
        ),
    }


def _build_context_builder_test_context() -> dict[str, Any]:
    """
    Representative Context Builder v2.0 output.
    """

    return {
        "schema_version": "2.0",

        "session": {
            "session_id": "mindo-pdf-integration-test",
            "start_time": (
                "2026-09-09T16:41:18.675782+00:00"
            ),
            "end_time": (
                "2026-09-09T16:42:19.703493+00:00"
            ),
            "duration_seconds": 61.027711,
            "total_turns": 4,
        },

        "conversation": {
            "turns": [
                {
                    "turn_id": 1,
                    "user": {
                        "text": "Hello.",
                        "key_points": [],
                    },
                    "mindo": {
                        "response_summary": (
                            "Asked an exploratory question."
                        ),
                    },
                    "emotion": {
                        "dominant": "neutral",
                        "supporting_signals": [
                            "happy",
                            "sad",
                        ],
                    },
                },
                {
                    "turn_id": 2,
                    "user": {
                        "text": (
                            "I am stressed of my exam tomorrow."
                        ),
                        "key_points": [
                            "concern:stress",
                            "concern:academic",
                            "situation:imminent_deadline",
                            "situation:academic_situation",
                        ],
                    },
                    "mindo": {
                        "response_summary": (
                            "Asked an exploratory question."
                        ),
                    },
                    "emotion": {
                        "dominant": "happy",
                        "supporting_signals": [
                            "neutral",
                            "fear",
                        ],
                    },
                },
                {
                    "turn_id": 3,
                    "user": {
                        "text": (
                            "um nothing but I am just scared."
                        ),
                        "key_points": [
                            "concern:fear",
                        ],
                    },
                    "mindo": {
                        "response_summary": (
                            "Asked a question."
                        ),
                    },
                    "emotion": {
                        "dominant": "fear",
                        "supporting_signals": [
                            "neutral",
                            "happy",
                        ],
                    },
                },
                {
                    "turn_id": 4,
                    "user": {
                        "text": "Okay, bye.",
                        "key_points": [],
                    },
                    "mindo": {
                        "response_summary": (
                            "Provided a conversational closing."
                        ),
                    },
                    "emotion": {
                        "dominant": "neutral",
                        "supporting_signals": [
                            "fear",
                            "happy",
                        ],
                    },
                },
            ]
        },

        "behavioral_signals": {
            "turn_observations": [
                {
                    "turn_id": 1,
                    "signals": [
                        {
                            "signal": "facial_emotion:neutral",
                            "source": "FER",
                            "evidence": (
                                "1 FER observation(s) "
                                "associated with this user turn."
                            ),
                            "confidence": "model_output",
                        },
                        {
                            "signal": "speech_duration",
                            "source": "browser_vad",
                            "evidence": (
                                "User speech duration: "
                                "1.255 seconds."
                            ),
                            "confidence": "observed",
                        },
                    ],
                }
            ]
        },

        "context": {
            "stated_concerns": [
                {
                    "text": (
                        "I am stressed of my exam tomorrow."
                    ),
                    "category": "stress",
                    "turns": [2],
                    "source": "conversation",
                },
                {
                    "text": (
                        "I am stressed of my exam tomorrow."
                    ),
                    "category": "academic",
                    "turns": [2],
                    "source": "conversation",
                },
                {
                    "text": (
                        "um nothing but I am just scared."
                    ),
                    "category": "fear",
                    "turns": [3],
                    "source": "conversation",
                },
            ],

            "situational_factors": [
                {
                    "text": (
                        "I am stressed of my exam tomorrow."
                    ),
                    "category": "imminent_deadline",
                    "turns": [2],
                    "source": "conversation",
                },
                {
                    "text": (
                        "I am stressed of my exam tomorrow."
                    ),
                    "category": "academic_situation",
                    "turns": [2],
                    "source": "conversation",
                },
            ],

            "observed_behavioral_patterns": [
                {
                    "signal": "short_user_response",
                    "description": (
                        "User response contains three "
                        "or fewer words."
                    ),
                    "turns": [1],
                    "source": "conversation",
                    "confidence": "observed",
                },
                {
                    "signal": "possible_topic_disengagement",
                    "description": (
                        "User response contains language "
                        "indicating possible disengagement "
                        "from the current topic."
                    ),
                    "turns": [3],
                    "source": "conversation",
                    "confidence": "observed",
                },
                {
                    "signal": "short_user_response",
                    "description": (
                        "User response contains three "
                        "or fewer words."
                    ),
                    "turns": [4],
                    "source": "conversation",
                    "confidence": "observed",
                },
            ],
        },

        "safety": {
            "assessment_status": "not_assessed",
        },
    }


def _run_pdf_only_test() -> None:
    """
    Fast test of PDF generation only.
    """

    print("\n==============================")
    print("MINDO PDF REPORT TEST")
    print("==============================")

    generator = PDFReportGenerator(
        output_directory="reports"
    )

    assessment = _build_test_assessment()

    output_path = generator.generate(
        assessment_result=assessment,
        filename="mindo_assessment_test.pdf",
    )

    print(
        "\nPDF generated successfully:"
    )

    print(
        output_path.resolve()
    )

    print(
        f"\nFile size: "
        f"{output_path.stat().st_size} bytes"
    )

    print("\n==============================")
    print("PDF REPORT TEST COMPLETE")
    print("==============================")


def _run_full_integration_test() -> None:
    """
    Test the real:

        Context Builder
            ↓
        MINDORAG
            ↓
        Assessment Engine
            ↓
        PDF Generator
    """

    print("\n==============================")
    print("MINDO FULL PDF INTEGRATION TEST")
    print("==============================")

    context = (
        _build_context_builder_test_context()
    )

    generator = PDFReportGenerator(
        output_directory="reports"
    )

    start_time = time.perf_counter()

    print("\nRunning:")

    print(
        "Context Builder v2.0"
    )

    print("        ↓")

    print("MINDO RAG")

    print("        ↓")

    print("Assessment Engine")

    print("        ↓")

    print("PDF Generator")

    print(
        "\nRunning full integration..."
    )

    output_path, assessment_result = (
        generator.generate_from_context(
            context=context
        )
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\nTotal integration time: "
        f"{total_time:.2f} sec"
    )

    print(
        "\n=============================="
    )

    print(
        "INTEGRATION RESULT"
    )

    print(
        "=============================="
    )

    assessment = assessment_result.get(
        "assessment",
        {},
    )

    if not isinstance(
        assessment,
        dict,
    ):
        assessment = {}

    print(
        f"\nCategory: "
        f"{assessment.get('category')}"
    )

    print(
        f"Confidence: "
        f"{assessment.get('confidence')}"
    )

    safety = assessment_result.get(
        "safety",
        {},
    )

    if not isinstance(
        safety,
        dict,
    ):
        safety = {}

    print(
        f"Safety: "
        f"{safety.get('status')}"
    )

    session = assessment_result.get(
        "session",
        {},
    )

    if not isinstance(
        session,
        dict,
    ):
        session = {}

    print(
        f"Session ID: "
        f"{session.get('session_id')}"
    )

    print(
        "\nPDF generated:"
    )

    print(
        output_path.resolve()
    )

    print(
        f"\nPDF size: "
        f"{output_path.stat().st_size} bytes"
    )

    print(
        "\n=============================="
    )

    print(
        "MINDO FULL PDF INTEGRATION TEST COMPLETE"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":

    try:

        # Fast PDF-only test
        _run_pdf_only_test()

        # Real Context Builder → RAG → Assessment → PDF test
        _run_full_integration_test()

    finally:

        # The RAG pipeline is shared and no longer closed per
        # request, so close it once when this local test ends.
        close_mindo_rag()