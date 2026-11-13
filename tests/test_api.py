from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_lab.api import app


@pytest.fixture
def client():
    # No lifespan: inject fake dependencies without reading .env or making network calls.
    app.state.rag = MagicMock()
    client = TestClient(app)
    yield client
    client.close()


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "question", "top_k": 0},
        {"query": "question", "top_k": 11},
        {"query": "question", "mode": "unknown"},
    ],
)
def test_invalid_inputs_return_422(client, body):
    assert client.post("/api/v1/search", json=body).status_code == 422
    app.state.rag.search.assert_not_called()


def test_health_failure_is_http_503(client):
    app.state.rag.health.return_value = {"status": "error", "services": {}}
    assert client.get("/health").status_code == 503


def test_dependency_failure_is_sanitized(client):
    app.state.rag.search.side_effect = httpx.ConnectError("https://user:secret@host")
    response = client.post("/api/v1/search", json={"query": "backup"})
    assert response.status_code == 503
    assert response.json()["error_type"] == "ConnectError"
    assert "secret" not in response.text


def test_search_returns_canonical_excerpts(client, hit):
    app.state.rag.search.return_value = [hit]
    response = client.post("/api/v1/search", json={"query": "  backup  ", "mode": "keyword"})
    assert response.status_code == 200
    assert response.json()[0]["source"] == hit.source
    assert app.state.rag.search.call_args.args[0].query == "backup"


def test_openapi_documents_search_and_answer_requests(client):
    schema = client.get("/openapi.json").json()
    assert "/api/v1/search" in schema["paths"]
    assert "agentic" in schema["components"]["schemas"]["AskRequest"]["properties"]
