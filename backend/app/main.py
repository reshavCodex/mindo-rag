import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.app.api.reports import (
    router as reports_router,
)

from backend.app.api.chat import (
    router as chat_router,
)

from backend.app.rag.mindo_rag import (
    close_mindo_rag,
    get_mindo_rag,
    is_mindo_rag_ready,
)


# ============================================================
# RAG LIFECYCLE
#
# The MINDO RAG pipeline (knowledge-base load, chunking, BM25
# index, Qdrant / Gemini / Cohere clients) is expensive to
# build, so it is built ONCE when the service starts and
# closed ONCE when the service shuts down.
#
# The build runs in the BACKGROUND so the HTTP server can start
# listening immediately. This keeps Render's port detection and
# health checks working even while the knowledge base is still
# being parsed.
#
# If a request arrives before the build has finished, it simply
# waits for the SAME build (get_mindo_rag() is lock-guarded);
# it never starts a second one.
# ============================================================

async def _warm_up_rag(app: FastAPI) -> None:
    """
    Build the shared RAG pipeline in a worker thread.
    """

    print("[RAG LIFECYCLE] Warm-up started.")

    try:

        await asyncio.to_thread(get_mindo_rag)

        app.state.rag_warmup_error = None

        print("[RAG LIFECYCLE] Warm-up complete. RAG is ready.")

    except Exception as error:

        # The server stays up. Nothing is cached on failure, so
        # the next call to get_mindo_rag() will retry the build.
        app.state.rag_warmup_error = repr(error)

        print(
            "[RAG LIFECYCLE] Warm-up FAILED:",
            repr(error),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):

    # ---------------- STARTUP ----------------

    app.state.rag_warmup_error = None

    warmup_task = asyncio.create_task(
        _warm_up_rag(app)
    )

    app.state.rag_warmup_task = warmup_task

    try:

        yield

    finally:

        # ---------------- SHUTDOWN ----------------

        print("[RAG LIFECYCLE] Shutting down...")

        # If the warm-up is still building, closing waits for
        # that build to finish and then closes the result, so
        # nothing is left open.
        try:

            await asyncio.to_thread(close_mindo_rag)

        except Exception as error:

            print(
                "[RAG LIFECYCLE] Error during shutdown:",
                repr(error),
            )

        if not warmup_task.done():

            try:
                await warmup_task
            except Exception:
                pass

        print("[RAG LIFECYCLE] Shutdown complete.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MINDO RAG Backend",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# REPORT ROUTER
# ============================================================

app.include_router(
    reports_router
)


# ============================================================
# CHAT ROUTER
# ============================================================

app.include_router(
    chat_router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "MINDO RAG Backend",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
#
# Liveness only. It answers immediately, even while the RAG
# pipeline is still warming up, so Render does not mistake a
# slow warm-up for a dead service.
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


# ============================================================
# READY
#
# Readiness: 200 once the shared RAG pipeline is built,
# 503 while it is still warming up (or if the warm-up failed).
# ============================================================

@app.get("/ready")
def ready():

    if is_mindo_rag_ready():

        return {
            "status": "ready",
        }

    error = getattr(
        app.state,
        "rag_warmup_error",
        None,
    )

    body = {
        "status": (
            "failed"
            if error
            else "warming_up"
        ),
    }

    if error:
        body["detail"] = error

    return JSONResponse(
        status_code=503,
        content=body,
    )