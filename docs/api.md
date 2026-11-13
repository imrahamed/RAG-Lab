# API reference

**Retrieve evidence or ask a grounded question.**

[Documentation](README.md) · [Setup](getting-started.md) · [Architecture](architecture.md)

Start the API from the project root:

```powershell
uv run uvicorn rag_lab.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation, or `http://127.0.0.1:8000/openapi.json` for the generated contract. Ingestion and reindexing remain CLI operations. This API has no authentication or tenant isolation; retain the loopback binding for local use.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | Dependency checks and tracing state |
| GET | `/api/v1/documents` | Most recent 100 canonical document records |
| POST | `/api/v1/search` | A ranked list of active SQL-backed chunks |
| POST | `/api/v1/ask` | An answer, supplied evidence, workflow steps, latency, and trace ID |

## Search request

```json
{
  "query": "Who approves a production restore?",
  "mode": "hybrid",
  "top_k": 5
}
```

| Field | Contract |
| --- | --- |
| `query` | Required nonblank string, up to 4,000 characters; surrounding whitespace is stripped |
| `mode` | `keyword`, `semantic`, or `hybrid`; defaults to `hybrid` |
| `top_k` | Integer from 1 to 10; defaults to 5 |

Each hit includes `chunk_id`, `document_id`, `title`, `source`, `position`, `text`, and `score`. Chunk positions are zero-based. The source is a file URI or arXiv URL. Scores order results within a mode; BM25, vector, and RRF scores are not interchangeable probabilities. Empty retrieval returns `[]`.

## Answer request

Use the same fields and add `agentic`:

```powershell
$body = @{
    query = "Who approves a production restore?"
    mode = "hybrid"
    top_k = 5
    agentic = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/ask `
    -Method Post -ContentType "application/json" -Body $body
```

`agentic` defaults to `false`. When true, the workflow grades relevance and may rewrite and retry retrieval once. The response is complete JSON rather than a token stream; every question is independent.

| Response field | Meaning |
| --- | --- |
| `answer` | Generated text with requested numbered citations, or an abstention |
| `sources` | Exact excerpts supplied to the answer model; each adds a one-based `citation` to the hit fields |
| `steps` | Executed actions and brief operational details, without private model reasoning |
| `model` | Configured Ollama chat model |
| `elapsed_ms` | Pipeline time in milliseconds |
| `trace_id` | Locally created Langfuse trace ID, or `null` when unavailable |

Sources show the evidence made available to the model. They do not certify that every generated claim is supported or that every source was cited. Check each citation against its excerpt. A trace ID does not prove remote event delivery; verify the trace in Langfuse.

## Health and errors

| HTTP status | Meaning | Next action |
| --- | --- | --- |
| `200` | Successful request or healthy required dependencies | Inspect returned evidence or service state |
| `422` | Invalid request body, mode, query, or top-k | Correct the request using `/docs` |
| `503` | A dependency failed, or a required health check failed | Run `uv run rag doctor` and consult [operations](operations.md) |

Dependency errors are reported by class rather than raw connection messages, which can contain credentials. `/health` reports tracing state without a network authentication probe. `rag doctor` checks enabled Langfuse authentication and includes failure in its exit status.
