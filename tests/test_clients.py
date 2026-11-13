import json

import httpx
import pytest

from rag_lab.ollama import Ollama
from rag_lab.search import SearchIndex, reciprocal_rank_fusion


def use_transport(client, handler):
    client.client.close()
    client.client = httpx.Client(
        base_url="https://mock.example", transport=httpx.MockTransport(handler)
    )


def test_rrf_rewards_agreement_and_does_not_double_count_duplicates():
    fused = reciprocal_rank_fusion(
        [
            [{"_id": "a"}, {"_id": "a"}, {"_id": "b"}],
            [{"_id": "b"}, {"_id": "c"}],
        ]
    )
    assert fused[0]["_id"] == "b"
    assert next(hit["_score"] for hit in fused if hit["_id"] == "a") == pytest.approx(1 / 61)
    assert len(fused) == 3


def test_ollama_applies_correct_task_prefix_and_disables_truncation(settings):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    client = Ollama(settings)
    use_transport(client, handler)
    try:
        client.embed(["document"])
        client.embed(["question"], purpose="query")
        assert calls[0]["input"] == ["search_document: document"]
        assert calls[1]["input"] == ["search_query: question"]
        assert calls[0]["truncate"] is False
    finally:
        client.close()


@pytest.mark.parametrize("vectors", [[], [[1, 2]], [[0, 0, 0]], [[1, 2, 3], [1, 2, 3]]])
def test_bad_embedding_count_shape_or_norm_fails(settings, vectors):
    client = Ollama(settings)
    use_transport(client, lambda _: httpx.Response(200, json={"embeddings": vectors}))
    try:
        with pytest.raises(ValueError):
            client.embed(["text"])
    finally:
        client.close()


def test_bulk_partial_failure_is_not_silently_accepted(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"errors": True, "items": []})

    client = SearchIndex(settings)
    use_transport(client, handler)
    try:
        with pytest.raises(RuntimeError, match="bulk indexing failed"):
            client.write([{"id": "c1", "text": "some text"}])
        assert calls[0].content.endswith(b"\n")
        assert calls[0].url.params["refresh"] == "wait_for"
    finally:
        client.close()


def test_same_dimensions_with_different_model_is_rejected(settings):
    profile = {**settings.embedding_profile, "model": "different-model"}
    client = SearchIndex(settings)
    use_transport(
        client,
        lambda _: httpx.Response(
            200,
            json={
                settings.opensearch_index: {"mappings": {"_meta": {"embedding": profile}}},
            },
        ),
    )
    try:
        with pytest.raises(ValueError, match="embedding profile"):
            client.validate_profile()
    finally:
        client.close()


@pytest.mark.parametrize(
    "response",
    [
        {"timed_out": True},
        {"_shards": {"failed": 1}},
    ],
)
def test_incomplete_search_results_fail(response):
    with pytest.raises(RuntimeError, match="incomplete"):
        SearchIndex.hits(response)


def test_cleanup_failure_is_reported_even_with_http_200(settings):
    client = SearchIndex(settings)
    use_transport(client, lambda _: httpx.Response(200, json={"failures": [{"cause": "failure"}]}))
    try:
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            client.cleanup("doc", "active-revision")
    finally:
        client.close()
