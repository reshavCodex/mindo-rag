from typing import Any, Dict, List


class ContextBuilder:
    """
    Builds a clean unified context for the MINDO RAG pipeline.

    Input:
        Structured conversation + behavioral signals

    Output:
        Unified, non-diagnostic context

    Important:
        This component does NOT diagnose the user.
        Facial-expression outputs are treated only as
        model-generated behavioral signals.
    """

    def build(
        self,
        source_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert Context Builder source data into the
        normalized MINDO context contract.
        """

        if not isinstance(source_data, dict):
            raise ValueError(
                "source_data must be a dictionary."
            )

        session = source_data.get(
            "session",
            {},
        )

        conversation = source_data.get(
            "conversation",
            {},
        )

        behavioral_signals = source_data.get(
            "behavioral_signals",
            {},
        )

        context = source_data.get(
            "context",
            {},
        )

        safety = source_data.get(
            "safety",
            {
                "assessment_status": "not_assessed"
            },
        )

        return {
            "schema_version": "2.0",

            "session": self._build_session(
                session
            ),

            "conversation": self._build_conversation(
                conversation
            ),

            "behavioral_signals": (
                self._build_behavioral_signals(
                    behavioral_signals
                )
            ),

            "context": self._build_context(
                context
            ),

            "safety": self._build_safety(
                safety
            ),
        }

    @staticmethod
    def _build_session(
        session: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Keep only session-level information required
        downstream.
        """

        return {
            "session_id": session.get(
                "session_id"
            ),
            "duration_seconds": session.get(
                "duration_seconds"
            ),
        }

    @staticmethod
    def _build_conversation(
        conversation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize conversation turns.
        """

        turns = conversation.get(
            "turns",
            [],
        )

        normalized_turns = []

        for turn in turns:

            normalized_turns.append(
                {
                    "turn_number": turn.get(
                        "turn_number"
                    ),
                    "user_text": turn.get(
                        "user_text",
                        "",
                    ),
                    "assistant_text": turn.get(
                        "assistant_text",
                        "",
                    ),
                }
            )

        return {
            "turns": normalized_turns
        }

    @staticmethod
    def _build_behavioral_signals(
        behavioral_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize turn-level facial/model signals.

        These remain observations from the model.
        They are NOT interpreted as diagnoses.
        """

        observations = behavioral_signals.get(
            "turn_observations",
            [],
        )

        normalized_observations = []

        for observation in observations:

            normalized_observations.append(
                {
                    "turn_number": observation.get(
                        "turn_number"
                    ),
                    "dominant_expression": (
                        observation.get(
                            "dominant_expression"
                        )
                    ),
                    "probabilities": observation.get(
                        "probabilities",
                        {},
                    ),
                    "observation_count": (
                        observation.get(
                            "observation_count",
                            0,
                        )
                    ),
                }
            )

        return {
            "turn_observations": (
                normalized_observations
            )
        }

    @staticmethod
    def _build_context(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize contextual observations.

        No diagnostic conclusions are created here.
        """

        return {
            "stated_concerns": ContextBuilder._clean_list(
                context.get(
                    "stated_concerns",
                    [],
                )
            ),

            "situational_factors": (
                ContextBuilder._clean_list(
                    context.get(
                        "situational_factors",
                        [],
                    )
                )
            ),

            "observed_behavioral_patterns": (
                context.get(
                    "observed_behavioral_patterns",
                    [],
                )
            ),
        }

    @staticmethod
    def _build_safety(
        safety: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preserve safety status without creating a
        safety conclusion that was not actually assessed.
        """

        return {
            "assessment_status": safety.get(
                "assessment_status",
                "not_assessed",
            )
        }

    @staticmethod
    def _clean_list(
        values: Any,
    ) -> List[str]:
        """
        Keep only non-empty string values.
        """

        if not isinstance(values, list):
            return []

        return [
            value.strip()
            for value in values
            if isinstance(value, str)
            and value.strip()
        ]


if __name__ == "__main__":

    sample_input = {
        "schema_version": "2.0",

        "session": {
            "session_id": "demo-session",
            "duration_seconds": 60.7,
        },

        "conversation": {
            "turns": [
                {
                    "turn_number": 1,
                    "user_text": (
                        "I am stressed about "
                        "my exam tomorrow."
                    ),
                    "assistant_text": (
                        "What about the exam "
                        "is making you stressed?"
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
                        "fear": 0.339,
                        "neutral": 0.416,
                        "happy": 0.034,
                    },
                    "observation_count": 3,
                }
            ]
        },

        "context": {
            "stated_concerns": [
                "Exam tomorrow",
                "Stress related to exam",
            ],
            "situational_factors": [
                "Upcoming examination",
            ],
            "observed_behavioral_patterns": [
                {
                    "signal": (
                        "elevated fear probability"
                    ),
                    "source": (
                        "facial_expression_model"
                    ),
                    "turns": [1],
                    "confidence": "model_output",
                }
            ],
        },

        "safety": {
            "assessment_status": "not_assessed"
        },
    }

    builder = ContextBuilder()

    unified_context = builder.build(
        sample_input
    )

    print("\n==============================")
    print("MINDO CONTEXT BUILDER")
    print("==============================")

    print(
        f"\nSchema: "
        f"{unified_context['schema_version']}"
    )

    print(
        f"Session: "
        f"{unified_context['session']}"
    )

    print(
        f"Conversation turns: "
        f"{len(unified_context['conversation']['turns'])}"
    )

    print(
        f"Behavioral observations: "
        f"{len(unified_context['behavioral_signals']['turn_observations'])}"
    )

    print(
        f"Stated concerns: "
        f"{unified_context['context']['stated_concerns']}"
    )

    print(
        f"Safety status: "
        f"{unified_context['safety']['assessment_status']}"
    )