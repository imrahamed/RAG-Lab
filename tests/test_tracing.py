import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr, ValidationError

from rag_lab.config import Settings
from rag_lab.models import AskRequest
from rag_lab.ollama import Ollama
from rag_lab.pipeline import RagService
from rag_lab.tracing import Tracing


@pytest.fixture
def recorded_tracing():
    exporter = InMemorySpanExporter()
    tracing = Tracing()
    tracing.enabled = True
    tracing.client = Langfuse(
        public_key="pk-lf-test-" + uuid4().hex,
        secret_key="sk-lf-test",
        base_url="https://langfuse.example.com",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    yield tracing, exporter
    tracing.close()


def finished(tracing, exporter):
    tracing.client.flush()
    return exporter.get_finished_spans()


def test_disabled_tracing_does_not_initialize_sdk(monkeypatch, settings):
    create = MagicMock()
    monkeypatch.setattr(Tracing, "_create_client", create)
    tracing = Tracing(settings)
    with tracing.span("disabled") as span:
        assert span.trace_id is None
        span.update(output="ignored")
    tracing.close()
    assert tracing.status() == {"status": "disabled"}
    create.assert_not_called()


def test_enabled_tracing_requires_both_keys(settings):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**settings.model_dump(), "langfuse_enabled": True})


def test_sdk_initialization_error_is_isolated(monkeypatch, settings, caplog):
    monkeypatch.setattr(Tracing, "_create_client", MagicMock(side_effect=RuntimeError("secret")))
    configured = settings.model_copy(
        update={
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": SecretStr("sk-test"),
        }
    )
    tracing = Tracing(configured)
    with tracing.span("still-runs"):
        pass
    assert tracing.status()["status"] == "error"
    assert "secret" not in caplog.text


def test_nested_spans_share_trace_and_have_correct_parent(recorded_tracing):
    tracing, exporter = recorded_tracing
    with tracing.span("parent") as parent:
        with tracing.span("child") as child:
            assert parent.trace_id == child.trace_id
    spans = {span.name: span for span in finished(tracing, exporter)}
    assert spans["child"].parent.span_id == spans["parent"].context.span_id
    assert spans["parent"].parent is None


def test_text_is_omitted_by_default_but_metadata_is_recorded(recorded_tracing):
    tracing, exporter = recorded_tracing
    with tracing.span("private", input="private-question", metadata={"count": 2}) as span:
        span.update(output="private-answer", metadata={"result_count": 1})
    attributes = dict(finished(tracing, exporter)[0].attributes)
    assert "langfuse.observation.input" not in attributes
    assert "langfuse.observation.output" not in attributes
    assert "private-question" not in json.dumps(attributes)
    assert any("result_count" in key for key in attributes)


def test_capture_content_opt_in_records_inputs_and_outputs(recorded_tracing):
    tracing, exporter = recorded_tracing
    tracing.capture_content = True
    with tracing.span("visible", input="question") as span:
        span.update(output="answer")
    attributes = dict(finished(tracing, exporter)[0].attributes)
    assert "question" in attributes["langfuse.observation.input"]
    assert "answer" in attributes["langfuse.observation.output"]


def test_application_errors_propagate_with_sanitized_error_observations(recorded_tracing):
    tracing, exporter = recorded_tracing
    with pytest.raises(ValueError, match="private-connection-string"):
        with tracing.span("failed"):
            raise ValueError("private-connection-string")
    span = finished(tracing, exporter)[0]
    assert span.attributes["langfuse.observation.level"] == "ERROR"
    assert span.attributes["langfuse.observation.status_message"] == "ValueError"
    assert "private-connection-string" not in json.dumps(dict(span.attributes))
    assert not span.events  # No exception auto-capture with sensitive messages/stack traces.


@pytest.mark.parametrize("failure", ["enter", "update", "exit", "shutdown"])
def test_telemetry_failures_do_not_break_application(failure):
    tracing = Tracing()
    tracing.client = MagicMock()
    manager = tracing.client.start_as_current_observation.return_value
    span = manager.__enter__.return_value
    if failure == "enter":
        manager.__enter__.side_effect = RuntimeError("telemetry failure")
    elif failure == "update":
        span.update.side_effect = RuntimeError("telemetry failure")
    elif failure == "exit":
        manager.__exit__.side_effect = RuntimeError("telemetry failure")
    else:
        tracing.client.shutdown.side_effect = RuntimeError("telemetry failure")
    calls = []
    with tracing.span("operation") as observation:
        calls.append("application-result")
        observation.update(metadata={"count": 1})
    tracing.close()
    assert calls == ["application-result"]


def test_ollama_generation_exports_usage_and_model(settings, recorded_tracing):
    tracing, exporter = recorded_tracing
    client = Ollama(settings, tracing)
    client.client.close()
    client.client = httpx.Client(
        base_url="http://ollama.example",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "message": {"content": "answer"},
                    "prompt_eval_count": 42,
                    "eval_count": 7,
                    "total_duration": 1000000,
                },
            )
        ),
    )
    try:
        assert client.chat([{"role": "user", "content": "private-question"}]) == "answer"
        attributes = dict(finished(tracing, exporter)[0].attributes)
        assert attributes["langfuse.observation.model.name"] == settings.ollama_chat_model
        usage = json.loads(attributes["langfuse.observation.usage_details"])
        assert usage == {"input": 42, "output": 7}
    finally:
        client.close()


def test_full_ask_trace_spans_retrieval_context_and_generation(settings, hit, recorded_tracing):
    tracing, exporter = recorded_tracing
    service = RagService(settings)
    service.tracing = tracing
    service.ollama.tracing = tracing
    service.index.tracing = tracing
    service.db.tracing = tracing
    service.index.client.close()
    service.ollama.client.close()

    def search_response(request):
        if request.url.path.endswith("_mapping"):
            return httpx.Response(
                200,
                json={
                    settings.opensearch_index: {
                        "mappings": {
                            "_meta": {"embedding": settings.embedding_profile},
                        }
                    }
                },
            )
        return httpx.Response(200, json={"hits": {"hits": [{"_id": hit.chunk_id, "_score": 1}]}})

    def ollama_response(request):
        if request.url.path.endswith("embed"):
            return httpx.Response(200, json={"embeddings": [[1, 2, 3]], "prompt_eval_count": 4})
        return httpx.Response(
            200,
            json={"message": {"content": "30 days [1]"}, "prompt_eval_count": 80, "eval_count": 6},
        )

    service.index.client = httpx.Client(
        base_url="https://search.example", transport=httpx.MockTransport(search_response)
    )
    service.ollama.client = httpx.Client(
        base_url="http://ollama.example", transport=httpx.MockTransport(ollama_response)
    )
    service.db._hydrate = MagicMock(return_value=[hit.model_dump(exclude={"score"})])
    try:
        answer = service.ask(AskRequest(query="How long are daily backups retained?"))
        spans = finished(tracing, exporter)
        names = {span.name for span in spans}
        assert names >= {
            "rag.ask",
            "rag.retrieve",
            "opensearch.bm25",
            "ollama.embed.query",
            "opensearch.knn",
            "postgres.hydrate",
            "rag.context",
            "ollama.generate",
        }
        assert len({span.context.trace_id for span in spans}) == 1
        root = next(span for span in spans if span.name == "rag.ask")
        assert answer.trace_id == format(root.context.trace_id, "032x")
        assert answer.answer == "30 days [1]"
    finally:
        service.close()


def test_file_ingestion_records_loading_chunking_and_commit(settings, tmp_path, recorded_tracing):
    tracing, exporter = recorded_tracing
    path = tmp_path / "notes.md"
    path.write_text("A short note about backups.", encoding="utf-8")
    service = RagService(settings)
    service.tracing = tracing
    service.ollama.close()
    service.index.close()
    service.ollama = MagicMock()
    service.index = MagicMock()
    service.ollama.embed.return_value = [[1, 2, 3]]
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None

    @contextmanager
    def lock(_):
        yield connection

    service.document_lock = lock
    try:
        result = service.ingest_file(path)
        spans = finished(tracing, exporter)
        assert result["status"] == "indexed"
        assert {span.name for span in spans} >= {
            "rag.ingest_file",
            "rag.load_file",
            "rag.ingest",
            "rag.chunk",
            "postgres.persist",
        }
        assert len({span.context.trace_id for span in spans}) == 1
    finally:
        service.close()


def test_concurrent_requests_keep_separate_trace_contexts(recorded_tracing):
    tracing, exporter = recorded_tracing
    barrier = Barrier(2)

    def request(number):
        with tracing.span(f"request-{number}") as parent:
            barrier.wait(timeout=5)
            with tracing.span(f"child-{number}") as child:
                assert child.trace_id == parent.trace_id
            return parent.trace_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        ids = list(executor.map(request, [1, 2]))
    assert len(set(ids)) == 2
    spans = {span.name: span for span in finished(tracing, exporter)}
    for number in (1, 2):
        parent = spans[f"request-{number}"]
        assert parent.parent is None
        assert spans[f"child-{number}"].parent.span_id == parent.context.span_id


def test_env_configuration_reaches_sdk_constructor(monkeypatch, settings):
    constructor = MagicMock()
    monkeypatch.setattr("langfuse.Langfuse", constructor)
    configured = settings.model_copy(
        update={
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-configured",
            "langfuse_secret_key": SecretStr("sk-configured"),
            "langfuse_base_url": "https://langfuse.example.com/",
            "langfuse_environment": "testing",
            "langfuse_timeout_seconds": 7,
        }
    )
    tracing = Tracing(configured)
    arguments = constructor.call_args.kwargs
    assert arguments["base_url"] == "https://langfuse.example.com"
    assert arguments["public_key"] == "pk-configured"
    assert arguments["secret_key"] == "sk-configured"
    assert arguments["environment"] == "testing"
    assert arguments["timeout"] == 7
    assert arguments["tracing_enabled"] is True
    tracing.close()


def test_authentication_failures_are_sanitized():
    tracing = Tracing()
    tracing.enabled = True
    tracing.client = MagicMock()
    tracing.client.auth_check.side_effect = RuntimeError("sk-lf-private")
    assert tracing.status(check_connection=True) == {
        "status": "error",
        "error_type": "RuntimeError",
    }
