# Configuration reference

**One `.env` file connects the application to your environment.**

[Documentation](README.md) · [Setup](getting-started.md) · [Operations](operations.md) · [Tracing](observability.md)

Run CLI commands from the project root. [`Settings`](../src/rag_lab/config.py) reads `.env`; process environment variables take precedence. The notebook resolves the project root and loads that same file. Real values belong in `.env`, while [`.env.example`](../.env.example) documents the public configuration shape.

## Remote storage

| Variable | Required / default | Meaning |
| --- | --- | --- |
| `POSTGRES_DSN` | Required | A libpq-compatible PostgreSQL connection URI |
| `OPENSEARCH_URL` | Required | Base URL of the hosted search service |
| `OPENSEARCH_USERNAME` | Optional | Basic-auth username; supply with a password |
| `OPENSEARCH_PASSWORD` | Optional | Basic-auth password; supply with a username |
| `OPENSEARCH_INDEX` | `rag-lab-v1` | Application index name; lowercase letters, digits, `_`, and `-`, beginning with a letter |
| `OPENSEARCH_CA_CERT` | Unset | Optional private CA PEM file; TLS verification remains enabled |

Example values below are placeholders:

```dotenv
POSTGRES_DSN=postgresql://rag_user:CHANGE_ME@postgres.example.com:5432/rag?sslmode=verify-full
OPENSEARCH_URL=https://search.example.com:9200
OPENSEARCH_USERNAME=rag_user
OPENSEARCH_PASSWORD=CHANGE_ME
OPENSEARCH_INDEX=rag-lab-v1
```

Percent-encode reserved characters in PostgreSQL URI passwords. `sslmode=verify-full` verifies the server certificate and hostname; add `sslrootcert` to the DSN for a private CA when your provider requires it. Use a dedicated database/schema and index. SQL schema selection can be configured through the connection's search path.

The application expects the names above; legacy `POSTGRES_URI` and `OPENSEARCH_URI` are not loaded as aliases. Keep OpenSearch credentials separate from its URL. Authentication supports basic auth or an authenticating proxy, without AWS SigV4 signing.

## Local Ollama

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Local inference endpoint |
| `OLLAMA_CHAT_MODEL` | `llama3.2:3b` | Answer, grading, and rewriting model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for both documents and questions |
| `OLLAMA_CONTEXT_TOKENS` | `8192` | Requested chat context length; larger contexts use more memory |
| `EMBEDDING_DIMENSIONS` | `768` | Expected output vector length; returned vectors are checked |
| `EMBED_DOCUMENT_PREFIX` | `"search_document: "` | Task prefix applied to document chunks |
| `EMBED_QUERY_PREFIX` | `"search_query: "` | Task prefix applied to questions |

The Nomic defaults use the task prefixes specified in its [model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). Other models may require different prefixes or empty strings. Preserve the trailing spaces inside quotes.

> **Model migration:** changing an embedding model, its weights, dimensions, or prefixes requires a new index and a complete reindex. Equal dimensions do not mean equal vector spaces. `EMBEDDING_DIMENSIONS` validates output; it does not resize vectors.

## Chunking and retrieval

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHUNK_SIZE` | `1400` | Maximum character window; minimum 100 |
| `CHUNK_OVERLAP` | `200` | Requested overlap in characters; smaller than chunk size |
| `EMBED_BATCH_SIZE` | `16` | Chunks per embedding request; range 1–128 |
| `RETRIEVAL_CANDIDATES` | `30` | Candidate pool per search branch; range 10–1000 |
| `MAX_CONTEXT_CHARS` | `12000` | Budget for serialized evidence JSON; minimum 1000 |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Timeout used by Ollama and OpenSearch HTTP clients |

Character counts are not token counts. The evidence budget excludes the question, system message, and generated response. Embedding truncation is disabled so oversized input fails visibly. Chunking changes create a new document revision when you ingest or reindex.

## Hosted Langfuse

| Variable | Default | Meaning |
| --- | --- | --- |
| `LANGFUSE_ENABLED` | `false` | Create and export observations |
| `LANGFUSE_PUBLIC_KEY` | Unset | Project public key; required when enabled |
| `LANGFUSE_SECRET_KEY` | Unset | Project secret key; required when enabled |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Project region or hosted-instance URL |
| `LANGFUSE_ENVIRONMENT` | `development` | Environment label attached to observations |
| `LANGFUSE_CAPTURE_CONTENT` | `false` | Include questions, prompts, excerpts, answers, and source paths |
| `LANGFUSE_TIMEOUT_SECONDS` | `5` | SDK request timeout; range 1–60 seconds |

Operational metadata, model usage, counts, and sanitized error types are captured when tracing is enabled, even with text capture off. The application uses SDK major version 5; see [hosted-server compatibility and setup](observability.md). No Langfuse server is started locally.

## Apply a configuration change

1. Save `.env` and restart the API or notebook kernel so long-lived clients reload it.
2. Run `uv run rag doctor`.
3. If embedding identity changed, finish the [index migration](operations.md#change-the-embedding-model) before serving queries with the new configuration.
