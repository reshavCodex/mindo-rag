# MINDO RAG Backend

> **Using the live app from a phone?** MINDO is not optimised for mobile screens yet, and visuals may look broken. Please use your browser's **Desktop site** option.

Retrieval-augmented analysis and PDF report generation for [MINDO](https://github.com/reshavCodex/mindo-frontend), an AI-assisted counselling and wellness platform for students and young adults.

**Public service:** https://mindo-rag.onrender.com (Render free tier, may sleep when idle)

> To try the full app, open this URL once to wake the service (along with the [conversation](https://mindo-conversation.onrender.com) and [chatbot](https://mindo-chat-backend.onrender.com) services), then visit https://mindo-frontend.vercel.app. This is only because of free-tier hosting.

---

## Purpose

After a counselling session, this service takes the session's semantic context, retrieves relevant evidence from a knowledge base, asks Gemini to analyse it, and returns an assessment together with a PDF report.

## Role in MINDO

```text
Conversation Backend → RAG Backend → PDF report (base64) → Conversation Backend → Supabase
```

The Conversation Backend sends the semantic context. This service returns the generated PDF as base64 inside JSON, plus a short assessment. It does **not** upload anything to Supabase; the Conversation Backend decodes the PDF and stores it. The frontend never calls this service directly.

## RAG Pipeline

```text
Context → Query Builder → Dense (Qdrant) + Sparse (BM25)
→ Reciprocal Rank Fusion → Cohere Rerank → Gemini Analysis
→ Assessment Engine → PDF Report
```

| Stage | Implementation |
|---|---|
| Knowledge base | PDF/TXT/MD files under `knowledge_base/`, read with `pypdf` |
| Chunking | Sentence-aware chunks, 1000 characters with 200 overlap, plus metadata enrichment |
| Dense retrieval | Gemini embeddings (`gemini-embedding-2`, 1536 dimensions) searched in Qdrant |
| Sparse retrieval | BM25 (`rank-bm25`), built in memory from the same chunks |
| Fusion | Reciprocal Rank Fusion (`k = 60`): top 20 dense + top 20 sparse, best 10 kept |
| Reranking | Cohere (`rerank-v4.0-fast`); the shared pipeline uses 20 candidates and returns the top 5 |
| Analysis | Gemini (`gemini-3.5-flash-lite`) returning structured JSON |
| Assessment | `AssessmentEngine` turns the analysis into category, confidence, summary and recommendations |
| Report | PDF generated with ReportLab |

### Knowledge base

`knowledge_base/` holds 46 PDFs in eight folders: `assessment`, `communication`, `conditions`, `interventions`, `reporting`, `safety`, `symptoms` and `vision`. The documents are third-party publications and remain the property of their respective publishers.

### Qdrant

| Setting | Value |
|---|---|
| Collection | `mindo_knowledge` |
| Vector size | 1536 |
| Distance | Cosine |
| Used for | Dense retrieval only (BM25 is separate and in-memory) |

If `QDRANT_URL` is set, the service connects to Qdrant Cloud (or any remote Qdrant) with `QDRANT_API_KEY`. Otherwise it falls back to local persistent Qdrant storage for development. The collection is created if missing, and startup fails if an existing collection has a different vector size. `scripts/migrate_local_qdrant_to_cloud.py` is included for moving a local store to Qdrant Cloud.

## Multi-User Architecture

The RAG pipeline is built **once** when the service starts, not for every request:

```text
Service starts
→ initialise RAG resources (background thread)
→ load knowledge base, chunk it, build BM25, connect Qdrant, Gemini, Cohere
→ keep resources alive
→ many users submit requests
→ each request runs its own query, retrieval, analysis and PDF generation
→ resources are closed when the service shuts down
```

This avoids re-parsing the knowledge base and rebuilding the retrieval pipeline for every user. Details from the code:

- The build runs in the background so the HTTP server starts listening immediately (important for Render port detection). A request that arrives early waits for the same build and never starts a second one.
- Each request keeps its data in local variables; the shared pipeline is only read while serving requests.
- Each report is written to its own temporary directory, deleted after the PDF bytes are read.
- Simultaneous report generations are capped by a semaphore (`MINDO_MAX_CONCURRENT_REPORTS`, default 4). Requests wait up to `MINDO_REPORT_QUEUE_TIMEOUT_SECONDS` (default 300), then receive `503` with `Retry-After`.
- On shutdown the service stops accepting new runs and waits up to 60 seconds for in-flight runs.

This design is not a claim of unlimited scalability, and no concurrency testing results are documented here.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service status |
| GET | `/health` | Liveness (answers even while warming up) |
| GET | `/ready` | `200` once the pipeline is built, `503` while warming up or if warm-up failed |
| POST | `/api/v1/reports/generate` | Generate assessment and PDF from a semantic context |
| POST | `/api/v1/chat/retrieve` | Retrieval only: evidence for a text query (no analysis, no PDF) |

### `POST /api/v1/reports/generate`

The body is the semantic context JSON. It must be non-empty and contain `session.session_id`. Abbreviated example of the shape used in this repo's test fixture:

```json
{
  "schema_version": "2.0",
  "session": { "session_id": "mindo-test-session", "duration_seconds": 90 },
  "conversation": {
    "turns": [
      { "turn_number": 1, "user_text": "...", "assistant_text": "..." }
    ]
  },
  "behavioral_signals": { "turn_observations": [ ... ] },
  "context": { "stated_concerns": [ ... ], "situational_factors": [ ... ] },
  "safety": { "assessment_status": "not_assessed" }
}
```

Response:

```json
{
  "status": "success",
  "session_id": "...",
  "report_filename": "mindo_assessment_<session_id>.pdf",
  "report_path": "...",
  "report_base64": "<base64 PDF>",
  "assessment": {
    "category": "...",
    "confidence": "...",
    "summary": "...",
    "recommendations": [ ... ]
  }
}
```

Errors: `400` for invalid input, `503` when the report queue is full, `500` for generation failures.

### `POST /api/v1/chat/retrieve`

Request `{ "query": "..." }`, response `{ "query": "...", "evidence": [ ... ] }`. It reuses the shared retrieval pipeline. The current chatbot service does not call it.

## Environment Variables

Names only. Never commit values. Locally the app loads `backend/.env`.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini embeddings and analysis (required) |
| `COHERE_API_KEY` | Cohere reranking (required) |
| `QDRANT_URL` | Qdrant Cloud / remote Qdrant URL (local fallback if unset) |
| `QDRANT_API_KEY` | Qdrant API key |
| `MINDO_MAX_CONCURRENT_REPORTS` | Concurrent report limit (default 4) |
| `MINDO_REPORT_QUEUE_TIMEOUT_SECONDS` | Max wait for a report slot (default 300) |
| `GEMINI_EMBEDDING_MODEL` | Declared in `app/config.py`; the embedding service currently uses its own default (`gemini-embedding-2`) |

## Local Setup and Running

```bash
git clone https://github.com/reshavCodex/mindo-rag.git
cd mindo-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Create `backend/.env` with the variables above, make sure a Qdrant collection named `mindo_knowledge` is available (see Qdrant), then run from the repository root:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
```

Check `http://127.0.0.1:8001/ready` until it returns `200` (the first start parses the knowledge base).

## Deployment

Deployed as a Render web service at https://mindo-rag.onrender.com, using Qdrant Cloud for vectors. Set the environment variables in the Render dashboard. After a cold start, expect a delay while the pipeline warms up; `/health` responds immediately and `/ready` reports when requests can be served.

The FastAPI app registers only the reports and chat routers. Other modules in the tree (such as `realtime/` and `vision/`) and the static `frontend/` folder are not mounted by `main.py`.

## Security Considerations

- API keys (Gemini, Cohere, Qdrant) come from the environment and must never be committed.
- The endpoints currently have no authentication of their own; anyone who knows the URL can call them, which consumes Gemini and Cohere quota. Treat the URL as an internal service endpoint.
- Session ids are sanitised before use in filenames, and each report uses an isolated temporary directory.
- Request content (queries, conversation text) is processed in memory and printed to logs in places; review logging before handling real user data.

## Part of the MINDO Project

| Repository | Role |
|---|---|
| [mindo-frontend](https://github.com/reshavCodex/mindo-frontend) | **Main project** and web app |
| [mindo-conversation](https://github.com/reshavCodex/mindo-conversation) | Calls this service after each session; stores the PDF in Supabase |
| [mindo-rag](https://github.com/reshavCodex/mindo-rag) | This repo |
| [mindo-chat-backend](https://github.com/reshavCodex/mindo-chat-backend) | Separate chatbot service |

Live app: https://mindo-frontend.vercel.app. For the full overview, read the [main MINDO README](https://github.com/reshavCodex/mindo-frontend).

## Disclaimer

MINDO is an AI-assisted wellness and academic project, not a replacement for professional mental-health care. Generated assessments are not diagnoses.

## Built By

**Reshav Pradhan**: [LinkedIn](https://www.linkedin.com/in/reshavpradhan/) · [GitHub](https://github.com/reshavCodex)

## License

No explicit open-source license has been selected. Until one is added, default copyright applies.
