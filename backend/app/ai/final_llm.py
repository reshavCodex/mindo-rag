from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.app.config import settings
from backend.app.ai.prompts import FINAL_LLM_SYSTEM_PROMPT


class FinalLLM:
    """
    Final LLM for the MINDO assessment pipeline.

    Responsibilities:
        1. Receive structured context from Context Builder.
        2. Receive evidence retrieved by the RAG pipeline.
        3. Synthesize observations using Gemini.
        4. Return predictable structured data for AssessmentEngine.

    Important:
        This class does NOT make the final diagnosis.
        AssessmentEngine remains responsible for the final
        assessment classification.
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 2

    REQUEST_TIMEOUT_MS = 60000
    MAX_OUTPUT_TOKENS = 1200

    REQUIRED_FIELDS = [
        "summary",
        "key_observations",
        "risk_indicators",
        "protective_factors",
        "supporting_evidence",
        "recommended_next_steps",
        "assessment_limitations",
    ]

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ) -> None:

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.model = model

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=self.REQUEST_TIMEOUT_MS
            ),
        )

    def _build_prompt(
        self,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Build the user-facing prompt.

        The system instruction is supplied separately through
        GenerateContentConfig, so it is intentionally NOT
        duplicated here.
        """

        context_json = json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        evidence_json = json.dumps(
            evidence,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return (
            "Analyze the following MINDO session context using "
            "only the supplied information and retrieved evidence.\n\n"
            "MINDO CONTEXT:\n"
            f"{context_json}\n\n"
            "RETRIEVED EVIDENCE:\n"
            f"{evidence_json}\n\n"
            "Return the assessment analysis as JSON only."
        )

    def _response_schema(self) -> dict[str, Any]:
        """
        JSON schema used to constrain Gemini's response.
        """

        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                },
                "key_observations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "risk_indicators": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "protective_factors": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "supporting_evidence": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "recommended_next_steps": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "assessment_limitations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": self.REQUIRED_FIELDS,
        }

    def _parse_response(
        self,
        response: Any,
    ) -> dict[str, Any]:
        """
        Validate and parse Gemini's structured JSON response.
        """

        text = response.text

        if not text or not text.strip():
            raise RuntimeError(
                "Gemini returned an empty Final LLM response."
            )

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Gemini returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                "Final LLM response must be a JSON object."
            )

        for field in self.REQUIRED_FIELDS:
            if field not in result:
                raise RuntimeError(
                    "Final LLM response is missing "
                    f"required field: {field}"
                )

        return result

    def _is_retryable_error(
        self,
        error: Exception,
    ) -> bool:
        """
        Determine whether a Gemini API error is likely transient.

        Retryable:
            429 - rate limit
            500 - internal server error
            502 - bad gateway
            503 - service unavailable
            504 - gateway timeout

        Other errors are allowed to propagate immediately.
        """

        if not isinstance(error, APIError):
            return False

        status_code = getattr(
            error,
            "code",
            None,
        )

        return status_code in {
            429,
            500,
            502,
            503,
            504,
        }

    def _generate(
        self,
        prompt: str,
    ) -> Any:
        """
        Generate the Final LLM response with controlled retries.
        """

        last_error: Exception | None = None

        for attempt in range(
            self.MAX_RETRIES + 1
        ):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=FINAL_LLM_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=self._response_schema(),
                        max_output_tokens=self.MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                    ),
                )

            except Exception as error:
                last_error = error

                if not self._is_retryable_error(
                    error
                ):
                    raise

                if attempt >= self.MAX_RETRIES:
                    raise

                delay = (
                    self.RETRY_DELAY_SECONDS
                    * (2 ** attempt)
                )

                print(
                    f"Final LLM transient error "
                    f"(attempt {attempt + 1}/"
                    f"{self.MAX_RETRIES + 1}). "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Final LLM request failed unexpectedly."
        )

    def analyze(
        self,
        context: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze the session context and retrieved evidence.
        """

        if not context:
            raise ValueError(
                "Context cannot be empty."
            )

        if evidence is None:
            evidence = []

        prompt = self._build_prompt(
            context=context,
            evidence=evidence,
        )

        response = self._generate(
            prompt=prompt,
        )

        return self._parse_response(
            response
        )


if __name__ == "__main__":
    print("\n==============================")
    print("MINDO FINAL LLM TEST")
    print("==============================")

    final_llm = FinalLLM()

    test_context = {
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
                        "That sounds difficult. "
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

    test_evidence = [
        {
            "text": (
                "Stress can affect sleep, concentration, "
                "and daily functioning."
            ),
            "source": (
                "caring_for_your_mental_health_nimh.pdf"
            ),
            "page_start": 1,
            "page_end": 1,
            "score": 0.80,
        },
        {
            "text": (
                "Getting enough sleep is important for "
                "overall wellbeing."
            ),
            "source": (
                "caring_for_your_mental_health_nimh.pdf"
            ),
            "page_start": 2,
            "page_end": 2,
            "score": 0.72,
        },
    ]

    print(
        f"\nModel: {final_llm.model}"
    )

    print(
        "Sending context + evidence to Gemini..."
    )

    start = time.perf_counter()

    result = final_llm.analyze(
        context=test_context,
        evidence=test_evidence,
    )

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"\nTime: {elapsed:.2f} seconds"
    )

    print("\n==============================")
    print("FINAL LLM RESULT")
    print("==============================")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n==============================")
    print("FINAL LLM TEST COMPLETE")
    print("==============================")