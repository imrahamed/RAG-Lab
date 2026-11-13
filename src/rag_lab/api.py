import logging
from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_lab.config import Settings
from rag_lab.models import Answer, AskRequest, Hit, SearchRequest
from rag_lab.pipeline import RagService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RagService(Settings())
    try:
        yield
    finally:
        app.state.rag.close()


app = FastAPI(
    title="RAG Lab",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "Remote PostgreSQL + OpenSearch, local Ollama. Ingest files with the CLI, "
        "then compare keyword, semantic and hybrid search here."
    ),
)


async def dependency_error(request: Request, error: Exception):
    logger.warning("RAG request failed (%s)", type(error).__name__)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "A RAG dependency failed. Run `rag doctor` and check service configuration.",
            "error_type": type(error).__name__,
        },
    )


for error_type in (httpx.HTTPError, psycopg.Error, RuntimeError, ValueError):
    app.add_exception_handler(error_type, dependency_error)


@app.get("/health")
def health(request: Request):
    result = request.app.state.rag.health()
    return JSONResponse(status_code=200 if result["status"] == "ok" else 503, content=result)


@app.get("/api/v1/documents")
def documents(request: Request):
    return request.app.state.rag.db.documents()


@app.post("/api/v1/search", response_model=list[Hit])
def search(body: SearchRequest, request: Request):
    return request.app.state.rag.search(body)


@app.post("/api/v1/ask", response_model=Answer)
def ask(body: AskRequest, request: Request):
    return request.app.state.rag.ask(body)
