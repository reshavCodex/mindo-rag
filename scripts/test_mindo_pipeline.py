import json

from backend.app.rag.mindo_rag import MINDORAG


context = {
    "schema_version": "2.0",

    "session": {
        "session_id": "test-session-001",
        "duration_seconds": 60.7
    },

    "conversation": {
        "turns": [
            {
                "turn_number": 1,
                "user_text": "I am stressed about my exam tomorrow.",
                "assistant_text": "What about the exam is making you feel stressed?"
            },
            {
                "turn_number": 2,
                "user_text": "I haven't been sleeping properly and I keep worrying about it.",
                "assistant_text": "How has this been affecting your daily routine?"
            }
        ]
    },

    "behavioral_signals": {
        "turn_observations": [
            {
                "turn_number": 1,
                "dominant_expression": "neutral",
                "probabilities": {
                    "neutral": 0.72,
                    "sad": 0.18,
                    "happy": 0.10
                },
                "observation_count": 3
            }
        ]
    },

    "context": {
        "stated_concerns": [
            "Stress related to exam",
            "Difficulty sleeping",
            "Worry about upcoming exam"
        ],
        "situational_factors": [
            "Exam tomorrow"
        ],
        "observed_behavioral_patterns": [
            {
                "signal": "Predominantly neutral facial expression",
                "source": "facial_expression_model",
                "turns": [1],
                "confidence": "model_output"
            }
        ]
    },

    "safety": {
        "assessment_status": "not_assessed"
    }
}


def main():

    print("\n" + "=" * 60)
    print("MINDO END-TO-END PIPELINE TEST")
    print("=" * 60)

    rag = MINDORAG()

    print("\nRunning pipeline...\n")

    result = rag.run(context)

    print("=" * 60)
    print("GENERATED RAG QUERY")
    print("=" * 60)

    print(result["query"])

    print("\n" + "=" * 60)
    print("RETRIEVED EVIDENCE")
    print("=" * 60)

    print(f"Evidence count: {len(result['evidence'])}")

    for index, evidence in enumerate(result["evidence"], start=1):
        print(f"\nEvidence {index}")

        print(
            "Score:",
            evidence.get("score")
        )

        print(
            "Source:",
            evidence.get("metadata", {}).get("source")
        )

        print(
            "Text:",
            evidence.get("text", "")[:300]
        )

        print("\n" + "=" * 60)
    print("FINAL LLM ANALYSIS")
    print("=" * 60)

    print(
        json.dumps(
            result["analysis"],
            indent=2,
            ensure_ascii=False
        )
    )

    print("\n" + "=" * 60)
    print("ASSESSMENT ENGINE OUTPUT")
    print("=" * 60)

    print(
        json.dumps(
            result["assessment"],
            indent=2,
            ensure_ascii=False
        )
    )

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()