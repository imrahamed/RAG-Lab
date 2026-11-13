import json
import ssl

import httpx

from rag_lab.config import Settings
from rag_lab.tracing import Tracing


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """Combine ranks, since BM25 and cosine-derived scores have different scales."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit["_id"]
            if key not in seen:
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                seen.add(key)
    return [
        {"_id": key, "_score": score}
        for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class SearchIndex:
    def __init__(self, settings: Settings, tracing: Tracing | None = None):
        self.settings = settings
        self.tracing = tracing or Tracing()
        self.index = settings.opensearch_index
        context = ssl.create_default_context(
            cafile=str(settings.opensearch_ca_cert) if settings.opensearch_ca_cert else None
        )
        auth = None
        if settings.opensearch_username:
            auth = (settings.opensearch_username, settings.opensearch_password.get_secret_value())
        self.client = httpx.Client(
            base_url=str(settings.opensearch_url).rstrip("/"),
            auth=auth,
            verify=context,
            timeout=settings.request_timeout_seconds,
        )

    def close(self):
        self.client.close()

    def request(self, method: str, path: str, **kwargs) -> dict:
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def validate_profile(self):
        mappings = self.request("GET", f"/{self.index}/_mapping")[self.index]["mappings"]
        if mappings.get("_meta", {}).get("embedding") != self.settings.embedding_profile:
            raise ValueError("Index embedding profile differs; use a new OPENSEARCH_INDEX")

    def initialize(self):
        response = self.client.head(f"/{self.index}")
        if response.status_code == 200:
            self.validate_profile()
            return
        if response.status_code != 404:
            response.raise_for_status()
        self.request(
            "PUT",
            f"/{self.index}",
            json={
                "settings": {
                    "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 1}
                },
                "mappings": {
                    "_meta": {"embedding": self.settings.embedding_profile},
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "revision": {"type": "keyword"},
                        "title": {"type": "text"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.settings.embedding_dimensions,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                            },
                        },
                    },
                },
            },
        )

    def health(self) -> dict:
        self.validate_profile()
        result = self.request("GET", f"/_cluster/health/{self.index}")
        if result["status"] == "red":
            raise RuntimeError("OpenSearch index is red")
        return {"status": "ok", "cluster_status": result["status"]}

    def write(self, chunks: list[dict]):
        with self.tracing.span("opensearch.index", metadata={"chunk_count": len(chunks)}):
            self._write(chunks)

    def _write(self, chunks: list[dict]):
        # Keep each bulk payload bounded. refresh=wait_for makes new chunks searchable.
        for start in range(0, len(chunks), 100):
            lines = []
            for chunk in chunks[start : start + 100]:
                chunk = chunk.copy()
                key = chunk.pop("id")
                lines.extend(
                    [json.dumps({"index": {"_index": self.index, "_id": key}}), json.dumps(chunk)]
                )
            result = self.request(
                "POST",
                "/_bulk",
                params={"refresh": "wait_for"},
                content="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
            if result.get("errors"):
                raise RuntimeError("OpenSearch bulk indexing failed; ingestion can be retried")

    def cleanup(self, document_id: str, active_revision: str):
        with self.tracing.span("opensearch.cleanup", metadata={"document_id": document_id}):
            self._cleanup(document_id, active_revision)

    def _cleanup(self, document_id: str, active_revision: str):
        result = self.request(
            "POST",
            f"/{self.index}/_delete_by_query",
            params={
                "refresh": "true",
                "conflicts": "proceed",
            },
            json={
                "query": {
                    "bool": {
                        "filter": [{"term": {"document_id": document_id}}],
                        "must_not": [{"term": {"revision": active_revision}}],
                    }
                }
            },
        )
        if result.get("failures") or result.get("version_conflicts") or result.get("timed_out"):
            raise RuntimeError("Search cleanup incomplete; run reindex")

    @staticmethod
    def hits(result: dict) -> list[dict]:
        if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("OpenSearch returned incomplete search results")
        return result["hits"]["hits"]

    def keyword(self, query: str, count: int) -> list[dict]:
        with self.tracing.span(
            "opensearch.bm25", as_type="retriever", input=query, metadata={"candidates": count}
        ) as span:
            hits = self._keyword(query, count)
            span.update(metadata={"hit_count": len(hits)})
            return hits

    def _keyword(self, query: str, count: int) -> list[dict]:
        result = self.request(
            "POST",
            f"/{self.index}/_search",
            json={
                "size": count,
                "_source": False,
                "query": {"multi_match": {"query": query, "fields": ["title^2", "text"]}},
            },
        )
        return self.hits(result)

    def semantic(self, vector: list[float], count: int) -> list[dict]:
        with self.tracing.span(
            "opensearch.knn",
            as_type="retriever",
            metadata={"candidates": count, "dimensions": len(vector)},
        ) as span:
            hits = self._semantic(vector, count)
            span.update(metadata={"hit_count": len(hits)})
            return hits

    def _semantic(self, vector: list[float], count: int) -> list[dict]:
        result = self.request(
            "POST",
            f"/{self.index}/_search",
            json={
                "size": count,
                "_source": False,
                "query": {"knn": {"embedding": {"vector": vector, "k": count}}},
            },
        )
        return self.hits(result)
