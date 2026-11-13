# Operations and troubleshooting

**Create application resources, verify dependencies, and rebuild derived data.**

[Documentation](README.md) · [Configuration](configuration.md) · [Tracing](observability.md)

## The command reference

| Command | Purpose |
| --- | --- |
| `uv run rag init` | Validate/create the configured search index and create application SQL tables |
| `uv run rag doctor` | Check storage, models, and enabled Langfuse authentication |
| `uv run rag documents` | List the latest 100 canonical documents |
| `uv run rag ingest data/sample` | Ingest supported files recursively; skip unchanged revisions |
| `uv run rag ingest data/sample --force` | Rebuild the selected local documents |
| `uv run rag arxiv 2005.11401` | Fetch and ingest one explicit arXiv PDF |
| `uv run rag reindex` | Rebuild every canonical document's search records from PostgreSQL |
| `uv run python scripts/evaluate.py --mode hybrid` | Run the small retrieval exercise after sample ingestion |

## Initialize an empty environment

```powershell
uv run rag init
uv run rag doctor
uv run rag ingest data/sample
```

`UndefinedTable` means a SQL query reached a database/schema without the expected table. OpenSearch HTTP 404 with `index_not_found_exception` means the named application index is absent. These can both occur when `doctor` runs before `init`. Initialize using the same `.env` as the API or notebook.

`init` is repeatable and does not delete application documents. It does not create remote servers, repair arbitrary schema drift, or migrate an incompatible vector field. If initialization stops partway through, fix the reported dependency and rerun it.

## Diagnose by symptom

| Symptom | Likely area | What to check |
| --- | --- | --- |
| Missing configuration | Environment | Preserve `.env`, fill required keys, run from the root |
| PostgreSQL connection error | Connectivity / TLS | DNS, VPN/firewall, port, database, URI encoding, root certificate |
| PostgreSQL `UndefinedTable` | Initialization / schema | Run `rag init`; confirm DSN and search path match |
| OpenSearch HTTP 404 | Index / endpoint | Confirm endpoint and index; initialize a missing application index |
| OpenSearch HTTP 401 or 403 | Authorization | Credentials and create/read/write/delete-by-query permissions |
| k-NN or mapping error | Search capability | Plugin, supported engine, dimension, and configured index |
| Embedding profile differs | Model identity | Use a fresh index and reindex every document |
| Ollama model/connection error | Local inference | Port 11434, local server, and both installed model names |
| Embedding input too long | Chunking | Reduce `CHUNK_SIZE`; reingest/reindex; truncation is disabled |
| Cleanup pending / too few hits | Derived index consistency | Run `reindex`; stale candidates may crowd out active chunks |
| Empty PDF text | Document parsing | OCR externally or ingest extracted Markdown/text |
| No Langfuse trace | Telemetry | Enable tracing, correct project region/keys, restart, authenticate, then inspect exporter/UI |

The current health response reports exception classes and does not expose every provider error detail. Use your provider's logs for the underlying reason. A generic `HTTPStatusError` alone does not distinguish missing indexes from authorization failures.

## Rebuild a missing or damaged search index

PostgreSQL retains extracted text and active chunks. The search index is derived data:

```powershell
uv run rag init
uv run rag reindex
uv run rag doctor
```

If the index exists with an incompatible mapping, choose a fresh index name instead of deleting shared infrastructure. Failed ingestion can leave orphan search records; canonical SQL hydration prevents them from being returned as evidence. Reindexing rewrites and cleans each canonical document's records. It does not globally purge orphan documents with no canonical SQL row.

## Change the embedding model

1. Pull the new local model and check its required prefixes and native output dimension.
2. Set the embedding model, dimensions, prefixes, and a fresh `OPENSEARCH_INDEX` in `.env`.
3. Run `uv run rag init`, then `uv run rag reindex` for every stored document.
4. Finish migration before serving queries with the new settings. Restart the API or notebook kernel.
5. Retire old indexes separately when you no longer need them.

Changing model weights behind an existing tag also changes the vector space. The application records the configured model name, dimensions, and prefixes, but does not fingerprint downloaded model weights.

## Document lifecycle

A source URI identifies a document. Reingestion of changed text atomically replaces its active SQL chunks after search indexing succeeds. Moving a local file creates a different identity. Removing a file does not remove stored data; automatic document deletion is not implemented.

PDF files need extractable text. Complex layouts, equations, and tables may require external parsing. arXiv ingestion uses the ID as the title and does not fetch a complete bibliographic record. Respect arXiv access policies; this project supplies no bulk crawler.

## Verify changes

```powershell
uv run ruff check .
uv run pytest
uv run --extra notebook python scripts/check_notebook.py
```

The notebook check executes offline and writes an HTML preview. Service-backed evaluation uses your actual configuration; the included three-question dataset is a learning exercise rather than a production quality benchmark. Schedule ingestion later with Task Scheduler or cron using the project directory as its working directory.
