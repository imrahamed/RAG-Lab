# Getting started

**From a configured environment to a cited answer.**

[Documentation](README.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Troubleshooting](operations.md)

## Before you begin

| Requirement | What the application needs |
| --- | --- |
| Python | Version 3.11 or later, with `uv` installed |
| PostgreSQL | An existing remote database; permission to create the two RAG tables and their indexes |
| OpenSearch | An existing remote 2.19+ or 3.x cluster with k-NN and Lucene HNSW support |
| Ollama | A local server with an embedding model and a chat model |
| Langfuse | Optional project keys for Langfuse Cloud or a compatible hosted instance |

The application creates its own tables and search index. It does not provision database servers or containers. PostgreSQL does not need `pgvector`.

## 1 · Install dependencies

Open PowerShell in the project directory:

```powershell
uv sync --extra dev
```

For the notebook, include its optional dependencies:

```powershell
uv sync --extra dev --extra notebook
```

If the default uv cache is inaccessible, use a project-local cache for the current terminal:

```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
```

## 2 · Configure your existing services

Preserve an existing `.env`; copy the example only when it is missing:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set `POSTGRES_DSN`, `OPENSEARCH_URL`, and your OpenSearch credentials. Keep `OLLAMA_URL` pointed at your local server. The [configuration reference](configuration.md) explains TLS certificates, model settings, and optional Langfuse tracing. Keep real credentials in `.env`, which is excluded from Git.

## 3 · Prepare local Ollama models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

If the installed Ollama application is not already serving, run `ollama serve`. Leave that terminal open and continue in another one. The default address is `http://127.0.0.1:11434`.

## 4 · Initialize before checking health

```powershell
uv run rag init
uv run rag doctor
```

`init` checks embedding dimensions, creates the named OpenSearch index, and creates `rag_documents` and `rag_chunks` in PostgreSQL. Repeating it validates the existing setup without deleting documents.

> **Checkpoint:** PostgreSQL, OpenSearch, and Ollama should report `status: ok`. With tracing enabled, `doctor` also checks Langfuse authentication. Before initialization, PostgreSQL can report `UndefinedTable` and OpenSearch can return HTTP 404 because the application's resources do not exist yet.

An OpenSearch status of green is ideal. Yellow can mean a replica has no eligible second node; red is treated as a failure.

## 5 · Ingest the sample corpus

```powershell
uv run rag ingest data/sample
uv run rag documents
```

The three Markdown files describe fictional incident response, database backup, and access policies. Each file produces an ingestion result with a document ID, status, and trace ID when tracing is active. Repeating the command skips unchanged documents.

## 6 · Retrieve, then ask

```powershell
uv run rag search "Who approves a production restore?" --mode keyword
uv run rag search "Who approves a production restore?" --mode semantic
uv run rag search "Who approves a production restore?" --mode hybrid
uv run rag ask "Who approves a production restore?"
```

> **Checkpoint:** inspect the backup-policy excerpt. The answer should identify the database operations lead and cite supporting evidence. Generated wording varies; the source text is what lets you check the claim.

Try `--agentic` to add relevance grading and, when necessary, one query rewrite and retrieval retry:

```powershell
uv run rag ask "What is the office cafeteria menu?" --agentic
```

The sample corpus contains no cafeteria policy. Inspect the returned steps and evidence rather than assuming fluent text is correct.

## 7 · Choose your interface

| Interface | Launch command | Best for |
| --- | --- | --- |
| CLI | `uv run rag --help` | Explicit ingestion, search, and operations |
| API | `uv run uvicorn rag_lab.api:app --host 127.0.0.1 --port 8000` | Requests from other applications; Swagger UI at `http://127.0.0.1:8000/docs` |
| Notebook | `uv run --extra notebook jupyter lab notebooks/rag_walkthrough.ipynb` | Learning the pipeline through executable experiments |

The API has no authentication or tenant isolation. Use the loopback binding for this learning project.

## Continue learning

Open the [notebook guide](../notebooks/README.md) for run modes, follow the [hands-on lessons](lessons.md), or inspect the [implementation](implementation.md) alongside the source code. If a dependency fails, use the [operations guide](operations.md) to isolate it.
