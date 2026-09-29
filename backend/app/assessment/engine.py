from __future__ import annotations

import json
from typing import Any


class AssessmentEngine:
    """
    MINDO Assessment Engine.

    Converts Final LLM observations into a structured,
    non-diagnostic assessment.

    The engine does NOT independently diagnose a mental
    health condition.
    """

    ENGINE_VERSION = "1.0"

    def assess(
        self,
        analysis: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        if not analysis:
            raise ValueError(
                "Analysis cannot be empty."
            )

        if not context:
            raise ValueError(
                "Context cannot be empty."
            )

        safety_status = self._get_safety_status(
            context
        )

        assessment_category = self._determine_category(
            analysis=analysis,
            safety_status=safety_status,
        )

        confidence = self._calculate_confidence(
            analysis=analysis,
            context=context,
        )

        return {
            "assessment_version": self.ENGINE_VERSION,

            "assessment": {
                "category": assessment_category,
                "confidence": confidence,
                "summary": analysis.get(
                    "summary",
                    "",
                ),
            },

            "observations": {
                "key_observations": analysis.get(
                    "key_observations",
                    [],
                ),
                "risk_indicators": analysis.get(
                    "risk_indicators",
                    [],
                ),
                "protective_factors": analysis.get(
                    "protective_factors",
                    [],
                ),
            },

            "evidence": {
                "supporting_evidence": analysis.get(
                    "supporting_evidence",
                    [],
                ),
            },

            "recommendations": analysis.get(
                "recommended_next_steps",
                [],
            ),

            "safety": {
                "status": safety_status,
            },

            "limitations": analysis.get(
                "assessment_limitations",
                [],
            ),

            "disclaimer": (
                "This assessment is an AI-generated, "
                "non-diagnostic summary based on the "
                "information available during the session. "
                "It is not a substitute for evaluation by "
                "a qualified healthcare professional."
            ),
        }

    def _get_safety_status(
        self,
        context: dict[str, Any],
    ) -> str:

        safety = context.get("safety", {})

        if not isinstance(safety, dict):
            return "not_assessed"

        status = safety.get(
            "assessment_status",
            "not_assessed",
        )

        if not status:
            return "not_assessed"

        return str(status)

    def _determine_category(
        self,
        analysis: dict[str, Any],
        safety_status: str,
    ) -> str:

        if safety_status.lower() in {
            "urgent",
            "high_risk",
            "critical",
        }:
            return "safety_concern"

        risk_indicators = analysis.get(
            "risk_indicators",
            [],
        )

        key_observations = analysis.get(
            "key_observations",
            [],
        )

        if risk_indicators:
            if key_observations:
                return "wellbeing_concern"

            return "potential_risk"

        if key_observations:
            return "general_wellbeing"

        return "insufficient_information"

    def _calculate_confidence(
        self,
        analysis: dict[str, Any],
        context: dict[str, Any],
    ) -> str:

        conversation = context.get(
            "conversation",
            {},
        )

        turns = []

        if isinstance(conversation, dict):
            turns = conversation.get(
                "turns",
                [],
            )

        observations = analysis.get(
            "key_observations",
            [],
        )

        evidence = analysis.get(
            "supporting_evidence",
            [],
        )

        if (
            len(turns) >= 5
            and len(observations) >= 3
            and len(evidence) >= 2
        ):
            return "high"

        if (
            len(turns) >= 2
            and len(observations) >= 1
            and len(evidence) >= 1
        ):
            return "moderate"

        return "low"


def _run_test() -> None:
    print("\n==============================")
    print("MINDO ASSESSMENT ENGINE TEST")
    print("==============================")

    engine = AssessmentEngine()

    context = {
        "schema_version": "2.0",
        "session": {
            "session_id": "test-session",
            "duration_seconds": 60,
        },
        "conversation": {
            "turns": [
                {
                    "turn_number": 1,
                    "user_text": (
                        "I have been stressed about my exams "
                        "and I haven't been sleeping properly."
                    ),
                    "assistant_text": (
                        "Can you tell me more about your sleep?"
                    ),
                }
            ]
        },
        "behavioral_signals": {
            "turn_observations": [
                {
                    "turn_number": 1,
                    "dominant_expression": "neutral",
                    "probabilities": {
                        "neutral": 0.70,
                        "sad": 0.15,
                        "angry": 0.05,
                        "happy": 0.10,
                    },
                    "observation_count": 3,
                }
            ]
        },
        "context": {
            "stated_concerns": [
                "exam stress",
                "sleep difficulty",
            ],
            "situational_factors": [
                "upcoming examinations",
            ],
            "observed_behavioral_patterns": [],
        },
        "safety": {
            "assessment_status": "not_assessed",
        },
    }

    analysis = {
        "summary": (
            "The user reported experiencing stress "
            "related to upcoming exams and associated "
            "sleep difficulties."
        ),
        "key_observations": [
            "User explicitly stated having exam stress.",
            "User explicitly reported sleep difficulty.",
        ],
        "risk_indicators": [
            "User-reported sleep difficulty associated "
            "with academic stress."
        ],
        "protective_factors": [],
        "supporting_evidence": [
            (
                "Retrieved evidence notes that stress "
                "can affect sleep."
            ),
            (
                "Retrieved evidence highlights the "
                "importance of sufficient sleep."
            ),
        ],
        "recommended_next_steps": [
            (
                "Consider general stress management "
                "and sleep hygiene practices."
            ),
            (
                "Consider speaking with a qualified "
                "healthcare professional or counselor."
            ),
        ],
        "assessment_limitations": [
            (
                "The interaction was brief and does not "
                "provide a complete psychological history."
            ),
        ],
    }

    result = engine.assess(
        analysis=analysis,
        context=context,
    )

    print("\n==============================")
    print("ASSESSMENT RESULT")
    print("==============================")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n==============================")
    print("ASSESSMENT ENGINE TEST COMPLETE")
    print("==============================")


if __name__ == "__main__":
    _run_test()