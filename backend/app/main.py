from fastapi import FastAPI

from backend.app.api.reports import (
    router as reports_router,
)

from backend.app.api.chat import (
    router as chat_router,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MINDO RAG Backend",
    version="1.0.0",
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
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
    }