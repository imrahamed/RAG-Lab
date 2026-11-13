import json
from unittest.mock import MagicMock

import pytest

from rag_lab.models import AskRequest, Hit, SearchRequest
from rag_lab.pipeline import RagService, build_context


@pytest.fixture
def service(settings):
    service = RagService(settings)
    service.ollama.close()
    service.index.close()
    service.ollama = MagicMock()
    service.index = MagicMock()
    service.db = MagicMock()
    return service


def test_context_budget_includes_escaping_and_matches_returned_excerpts(hit):
    hit = hit.model_copy(update={"text": '\\"\n' * 500})
    context, sources = build_context([hit], 300)
    assert len(context) <= 300
    assert sources and len(sources[0].text) < len(hit.text)
    assert json.loads(context)[0]["text"] == sources[0].text
    assert sources[0].citation == 1


def test_sources_omitted_when_metadata_alone_exceeds_budget(hit):
    context, sources = build_context([hit.model_copy(update={"title": "x" * 500})], 100)
    assert context == "[]"
    assert sources == []


def test_inactive_search_records_are_removed_using_canonical_sql(service, hit):
    service.index.keyword.return_value = [
        {"_id": "orphan", "_score": 99},
        {"_id": hit.chunk_id, "_score": 5},
    ]
    service.db.hydrate.return_value = {hit.chunk_id: hit.model_dump(exclude={"score"})}
    result = service.search(SearchRequest(query="backup", mode="keyword"))
    assert [item.chunk_id for item in result] == [hit.chunk_id]
    service.ollama.embed.assert_not_called()


def test_hybrid_uses_query_embedding_and_rank_fusion(service, hit):
    service.index.keyword.return_value = [{"_id": hit.chunk_id, "_score": 400}]
    service.index.semantic.return_value = [{"_id": hit.chunk_id, "_score": 0.8}]
    service.db.hydrate.return_value = {hit.chunk_id: hit.model_dump(exclude={"score"})}
    service.ollama.embed.return_value = [[1, 2, 3]]
    result = service.search(SearchRequest(query="backup"))
    service.ollama.embed.assert_called_once_with(["backup"], purpose="query")
    assert result[0].score == pytest.approx(2 / 61)


def test_empty_retrieval_abstains_without_generation(service):
    service.search = MagicMock(return_value=[])
    answer = service.ask(AskRequest(query="unsupported"))
    assert answer.sources == []
    assert answer.steps[-1].action == "abstain"
    service.ollama.chat.assert_not_called()


def test_agentic_workflow_has_at_most_two_retrievals(service):
    service.search = MagicMock(return_value=[])
    service.ollama.chat.return_value = "rewritten question"
    answer = service.ask(AskRequest(query="unsupported", agentic=True))
    assert service.search.call_count == 2
    assert service.ollama.chat.call_count == 1  # Rewrite only; no evidence to grade/generate.
    assert [step.action for step in answer.steps] == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "abstain",
    ]


def test_agentic_rewrite_preserves_original_question_for_generation(service, hit):
    service.search = MagicMock(side_effect=[[], [hit]])
    service.ollama.chat.side_effect = ["backup retention", '{"ids": [1]}', "30 days [1]"]
    answer = service.ask(AskRequest(query="How long are daily backups kept?", agentic=True))
    assert service.search.call_args_list[1].args[0].query == "backup retention"
    prompt = service.ollama.chat.call_args_list[-1].args[0][-1]["content"]
    assert "How long are daily backups kept?" in prompt
    assert answer.answer == "30 days [1]"
    assert len(answer.sources) == 1


@pytest.mark.parametrize("response", ['{"ids": "all"}', '{"ids": [true]}', "[]", "not json"])
def test_malformed_grading_fails_instead_of_bypassing_it(service, hit, response):
    service.ollama.chat.return_value = response
    with pytest.raises(ValueError):
        service._grade("question", [hit])


def test_indexing_failure_never_commits_new_sql_content(service):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    service.ollama.embed.return_value = [[1, 2, 3]]
    service.index.write.side_effect = RuntimeError("bulk failure")
    with pytest.raises(RuntimeError, match="bulk failure"):
        service._ingest(connection, "doc", "title", "source", "short document", False)
    connection.commit.assert_not_called()
    sql = [call.args[0] for call in connection.execute.call_args_list]
    assert not any("INSERT" in statement or "DELETE" in statement for statement in sql)


def test_cleanup_failure_keeps_successful_sql_commit_and_reports_recovery(service):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    service.ollama.embed.return_value = [[1, 2, 3]]
    service.index.cleanup.side_effect = RuntimeError("cleanup failure")
    result = service._ingest(connection, "doc", "title", "source", "short document", False)
    connection.commit.assert_called_once()
    assert result["status"] == "indexed"
    assert result["cleanup_pending"] is True


def test_unchanged_revision_skips_embedding_and_index_writes(service):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    service.ollama.embed.return_value = [[1, 2, 3]]
    service._ingest(connection, "doc", "title", "source", "short document", False)
    revision = service.index.write.call_args.args[0][0]["revision"]
    connection.execute.return_value.fetchone.return_value = {"revision": revision}
    service.ollama.reset_mock()
    service.index.reset_mock()
    result = service._ingest(connection, "doc", "title", "source", "short document", False)
    assert result["status"] == "unchanged"
    service.ollama.embed.assert_not_called()
    service.index.write.assert_not_called()


def test_health_does_not_echo_connection_credentials(service):
    service.db.health.side_effect = RuntimeError("postgresql://user:secret@example.com/db")
    service.index.health.return_value = {"status": "ok"}
    service.ollama.health.return_value = {"status": "ok"}
    report = service.health()
    assert report["status"] == "error"
    assert "secret" not in json.dumps(report)


def test_optional_tracing_failure_affects_doctor_but_not_api_health(service):
    service.db.health.return_value = {"status": "ok"}
    service.index.health.return_value = {"status": "ok"}
    service.ollama.health.return_value = {"status": "ok"}
    service.tracing = MagicMock()
    service.tracing.status.return_value = {"status": "error", "error_type": "RuntimeError"}
    assert service.health()["status"] == "ok"
    service.tracing.status.assert_called_with(check_connection=False)
    assert service.health(check_tracing=True)["status"] == "error"
    service.tracing.status.assert_called_with(check_connection=True)


def test_context_sources_have_contiguous_citation_numbers(hit):
    hits = [hit, Hit(**{**hit.model_dump(), "chunk_id": "c2", "position": 1})]
    _, sources = build_context(hits, 2000)
    assert [source.citation for source in sources] == [1, 2]
