import cohere

from app.config import settings


client = cohere.ClientV2(
    api_key=settings.cohere_api_key
)

query = "What helps improve mental health?"

documents = [
    "Regular exercise and sufficient sleep can support mental health.",
    "The capital of France is Paris.",
    "Staying connected with friends and family can provide emotional support.",
]

response = client.rerank(
    model="rerank-v4.0-fast",
    query=query,
    documents=documents,
    top_n=3,
)

print("\n==============================")
print("COHERE RERANK TEST")
print("==============================")

for rank, result in enumerate(
    response.results,
    start=1,
):
    print(
        f"\nRank: {rank}"
    )

    print(
        f"Index: {result.index}"
    )

    print(
        f"Score: {result.relevance_score:.6f}"
    )

    print(
        f"Document: {documents[result.index]}"
    )

print("\nCohere API is working.")