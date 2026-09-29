from __future__ import annotations

from typing import Any


class RAGQueryBuilder:
    """
    Builds a retrieval query from the Context Builder v2.0 output.

    The query is intentionally focused on:
    - stated concerns
    - situational factors
    - relevant recent user statements

    Irrelevant conversational closing messages are excluded.
    """

    MAX_QUERY_LENGTH = 1000
    MAX_RECENT_TURNS = 3

    # Common conversational messages that should not influence retrieval.
    IRRELEVANT_PHRASES = {
        "okay",
        "ok",
        "bye",
        "goodbye",
        "okay bye",
        "ok bye",
        "thanks",
        "thank you",
        "thank you bye",
        "see you",
        "see you later",
        "that's all",
        "thats all",
        "nothing",
        "nothing else",
    }

    def build_query(self, context: dict[str, Any]) -> str:
        """
        Build a retrieval query from Context Builder v2.0 output.
        """

        if not context:
            raise ValueError("Context cannot be empty.")

        parts: list[str] = []

        # ---------------------------------------------------------
        # 1. Explicitly stated concerns
        # ---------------------------------------------------------
        self._add_context_items(
            parts,
            context,
            "stated_concerns",
        )

        # ---------------------------------------------------------
        # 2. Situational factors
        # ---------------------------------------------------------
        self._add_context_items(
            parts,
            context,
            "situational_factors",
        )

        # ---------------------------------------------------------
        # 3. Relevant recent user statements
        # ---------------------------------------------------------
        self._add_recent_user_turns(
            parts,
            context,
        )

        # ---------------------------------------------------------
        # 4. Deduplicate
        # ---------------------------------------------------------
        query = self._deduplicate(parts)

        if not query:
            raise ValueError(
                "Unable to build a meaningful RAG query from context."
            )

        # ---------------------------------------------------------
        # 5. Protect retrieval system from excessively large query
        # ---------------------------------------------------------
        return query[: self.MAX_QUERY_LENGTH]

    def _add_context_items(
        self,
        parts: list[str],
        context: dict[str, Any],
        field_name: str,
    ) -> None:
        """
        Add structured context items such as:

        stated_concerns
        situational_factors
        """

        context_section = context.get("context", {})

        if not isinstance(context_section, dict):
            return

        items = context_section.get(field_name, [])

        if not isinstance(items, list):
            return

        for item in items:

            if isinstance(item, dict):

                text = item.get("text")
                category = item.get("category")

                if category:
                    category_text = str(category).strip()

                    if category_text:
                        parts.append(category_text)

                if text:
                    text_value = str(text).strip()

                    if text_value and not self._is_irrelevant_text(
                        text_value
                    ):
                        parts.append(text_value)

            elif isinstance(item, str):

                text_value = item.strip()

                if text_value and not self._is_irrelevant_text(
                    text_value
                ):
                    parts.append(text_value)

    def _add_recent_user_turns(
        self,
        parts: list[str],
        context: dict[str, Any],
    ) -> None:
        """
        Add relevant recent user statements.

        We intentionally avoid blindly adding every recent turn.
        Short conversational closings such as 'Okay, bye.' should
        not affect knowledge retrieval.
        """

        conversation = context.get("conversation", {})

        if not isinstance(conversation, dict):
            return

        turns = conversation.get("turns", [])

        if not isinstance(turns, list):
            return

        valid_turns = [
            turn
            for turn in turns
            if isinstance(turn, dict)
        ]

        recent_turns = valid_turns[-self.MAX_RECENT_TURNS :]

        for turn in recent_turns:

            user_text = self._extract_user_text(turn)

            if not user_text:
                continue

            if self._is_irrelevant_text(user_text):
                continue

            # Very short conversational responses generally have
            # little retrieval value unless they contain a meaningful
            # concern keyword.
            if self._is_low_information_short_text(user_text, turn):
                continue

            parts.append(user_text)

    def _extract_user_text(
        self,
        turn: dict[str, Any],
    ) -> str:
        """
        Extract user text from Context Builder v2.0.

        Expected structure:

        {
            "user": {
                "text": "..."
            }
        }

        Compatibility with older schemas is also retained.
        """

        user_data = turn.get("user")

        if isinstance(user_data, dict):

            text = user_data.get("text")

            if text:
                return str(text).strip()

        elif isinstance(user_data, str):

            return user_data.strip()

        # Backward compatibility
        direct_text = turn.get("user_text")

        if direct_text:
            return str(direct_text).strip()

        return ""

    def _is_irrelevant_text(self, text: str) -> bool:
        """
        Detect messages that should not influence retrieval.
        """

        normalized = " ".join(text.lower().split())

        if not normalized:
            return True

        if normalized in self.IRRELEVANT_PHRASES:
            return True

        return False

    def _is_low_information_short_text(
        self,
        text: str,
        turn: dict[str, Any],
    ) -> bool:
        """
        Filter very short responses only when they do not contain
        structured concerns or meaningful key points.

        Example:

        'bye'       -> filtered
        'okay'      -> filtered
        'scared'    -> retained
        'stress'    -> retained
        """

        words = text.split()

        if len(words) > 3:
            return False

        # If Context Builder identified key points, preserve the turn.
        user_data = turn.get("user")

        if isinstance(user_data, dict):

            key_points = user_data.get("key_points", [])

            if isinstance(key_points, list) and key_points:
                return False

        # Meaningful emotional / concern words should be retained.
        meaningful_keywords = {
            "stress",
            "stressed",
            "anxiety",
            "anxious",
            "fear",
            "scared",
            "sad",
            "depressed",
            "angry",
            "panic",
            "worried",
            "worry",
            "lonely",
            "sleep",
            "insomnia",
            "suicidal",
            "hopeless",
            "helpless",
        }

        normalized_words = {
            word.lower().strip(".,!?")
            for word in words
        }

        if normalized_words.intersection(meaningful_keywords):
            return False

        return True

    def _deduplicate(
        self,
        parts: list[str],
    ) -> str:
        """
        Remove duplicate text while preserving order.
        """

        seen: set[str] = set()
        unique_parts: list[str] = []

        for part in parts:

            normalized = " ".join(part.split()).lower()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_parts.append(part.strip())

        return " ".join(unique_parts)


def _build_test_context() -> dict[str, Any]:
    """
    Test Context Builder v2.0 payload.
    """

    return {
        "schema_version": "2.0",
        "session": {
            "session_id": "test-session",
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
                        "response_summary": "Asked an exploratory question."
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
                        "text": "I am stressed of my exam tomorrow.",
                        "key_points": [
                            "concern:stress",
                            "concern:academic",
                            "situation:imminent_deadline",
                            "situation:academic_situation",
                        ],
                    },
                    "mindo": {
                        "response_summary": "Asked an exploratory question."
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
                        "text": "um nothing but I am just scared.",
                        "key_points": [
                            "concern:fear",
                        ],
                    },
                    "mindo": {
                        "response_summary": "Asked a question."
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
                        "response_summary": "Provided a conversational closing."
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
            "turn_observations": []
        },
        "context": {
            "stated_concerns": [
                {
                    "text": "I am stressed of my exam tomorrow.",
                    "category": "stress",
                    "turns": [2],
                    "source": "conversation",
                },
                {
                    "text": "I am stressed of my exam tomorrow.",
                    "category": "academic",
                    "turns": [2],
                    "source": "conversation",
                },
                {
                    "text": "um nothing but I am just scared.",
                    "category": "fear",
                    "turns": [3],
                    "source": "conversation",
                },
            ],
            "situational_factors": [
                {
                    "text": "I am stressed of my exam tomorrow.",
                    "category": "imminent_deadline",
                    "turns": [2],
                    "source": "conversation",
                },
                {
                    "text": "I am stressed of my exam tomorrow.",
                    "category": "academic_situation",
                    "turns": [2],
                    "source": "conversation",
                },
            ],
            "observed_behavioral_patterns": [
                {
                    "signal": "short_user_response",
                    "description": "User response contains three or fewer words.",
                    "turns": [1],
                    "source": "conversation",
                    "confidence": "observed",
                },
                {
                    "signal": "possible_topic_disengagement",
                    "description": "User response contains language indicating possible disengagement from the current topic.",
                    "turns": [3],
                    "source": "conversation",
                    "confidence": "observed",
                },
            ],
        },
        "safety": {
            "assessment_status": "not_assessed"
        },
    }


def _run_test() -> None:

    builder = RAGQueryBuilder()

    context = _build_test_context()

    query = builder.build_query(context)

    print("=" * 70)
    print("RAG QUERY BUILDER TEST")
    print("=" * 70)

    print("\nGenerated RAG query:")
    print(query)

    print("\nQuery length:")
    print(len(query))

    print("\nContains 'Okay, bye.':")
    print("Okay, bye." in query)

    print("=" * 70)


if __name__ == "__main__":
    _run_test()