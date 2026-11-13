# Hands-on lessons

**Eight experiments to connect the concepts to working code.**

[Documentation](README.md) · [Setup](getting-started.md) · [Executable notebook](../notebooks/rag_walkthrough.ipynb) · [Implementation](implementation.md)

Use the README setup first. The commands below use the fictional Northstar sample documents.

| Stage | Lessons | Checkpoint |
| --- | --- | --- |
| Connect and ingest | 1–2 | Healthy services; unchanged documents are skipped |
| Retrieve and ground | 3–5 | Useful excerpts and claims you can verify |
| Adapt and recover | 6–7 | A bounded retry path and rebuildable search data |
| Observe | 8 | One operation connected by a trace ID |

## Lesson 1: Understand service boundaries

Run `uv run rag doctor`, then inspect `config.py` and `db.py`. Find which service owns each piece of state. PostgreSQL holds canonical text; OpenSearch holds a derived index; Ollama supplies computation. The application connects to existing servers and never starts containers.

Expected outcome: all three checks report `ok`. No credentials are needed by the test suite; actual ingestion needs your configured services.

## Lesson 2: Follow one document through ingestion

```powershell
uv run rag ingest data/sample/incident-response.md
uv run rag ingest data/sample/incident-response.md
```

The first run reports `indexed` and a chunk count. The second reports `unchanged`. Read `load_file`, `chunk_text`, and `_ingest` in that order. Change one sentence and run the command again; the document ID stays the same and its revision changes. Restore the sentence afterward.

Expected outcome: repeated ingestion does not accumulate duplicate active chunks. In PostgreSQL, `SELECT id, title, revision FROM rag_documents;` shows one source entry.

## Lesson 3: Establish the keyword baseline

```powershell
uv run rag ingest data/sample
uv run rag search "SEV-1 incident commander" --mode keyword
```

Read the `multi_match` body in `search.py`. Notice that this mode does not call Ollama for query embeddings. Inspect titles, text, positions, and scores. Exact identifiers make useful keyword test cases.

Expected outcome: `incident-response` appears among the results. An empty index or a nonmatching keyword query returns an empty list.

## Lesson 4: Compare semantic retrieval and hybrid fusion

```powershell
uv run rag search "How can I restore accidentally removed customer records?" --mode semantic
uv run rag search "How can I restore accidentally removed customer records?" --mode hybrid
```

The backups document should be useful even if the question uses different words. Trace the query embedding, the k-NN request, and `reciprocal_rank_fusion`. Do not compare raw keyword scores numerically with RRF scores: they mean different things.

Run `uv run python scripts/evaluate.py --mode keyword`, then repeat with `semantic` and `hybrid`. With three simple questions, ties and perfect results are unsurprising. Add harder queries before drawing conclusions.

## Lesson 5: Inspect evidence before trusting an answer

```powershell
uv run rag ask "How long are backups retained and who can approve a restore?"
uv run rag ask "What is the office cafeteria menu?" --mode keyword
```

Map every generated citation to its source excerpt. Find `build_context` and check how it handles the character budget. The unsupported keyword query should take the empty-evidence abstention path if it has no matching terms. Hybrid search can still find unrelated neighbors, so absence of relevant evidence is not the same as an empty result list.

Expected outcome: the backup answer is supported by the backup policy; unsupported questions are acknowledged, with no invented company policies.

## Lesson 6: Observe adaptive retrieval

```powershell
uv run rag ask "What is the office cafeteria menu?" --agentic
```

Inspect `steps`. If the grader rejects all excerpts, the system rewrites the query once, retrieves again, then abstains if still unsupported. Model decisions are not deterministic correctness guarantees, even with temperature 0. Find the two-attempt bound in the code.

## Lesson 7: Practice recovery

```powershell
uv run rag reindex
uv run rag doctor
```

Reindex reads canonical text from PostgreSQL and refreshes embeddings and OpenSearch records. For an embedding-model migration, choose a new index name first, run `init`, then reindex all documents. Finish migration before serving queries with the new configuration. Do not drop shared tables or remote indexes to perform this lesson.

For automatic ingestion later, schedule the same `uv run rag ingest <directory>` command with Windows Task Scheduler, using this project as its working directory. Keep the explicit CLI workflow while learning; no workflow scheduler is needed inside the application.

## Lesson 8: Inspect an end-to-end Langfuse trace

Configure a hosted Langfuse project using [the setup instructions](../README.md#enable-langfuse-tracing), then run:

```powershell
uv run rag doctor
uv run rag ask "Who approves a production restore?" --agentic
uv run rag ingest data/sample --force
```

Find the returned trace IDs in Langfuse. For the question, expand retrieval, embeddings, grading, context, and generation; compare their timing and token usage. Ingestion creates separate traces with parsing, chunking, indexing, and SQL persistence. Default traces omit text; enable content capture only when you want questions, prompts, evidence, and answers exported.

Expected outcome: nested steps share their parent's trace ID, token usage matches Ollama's response counters, and an unsupported question can show grading, one rewrite, and abstention. See [the tracing walkthrough](observability.md) for the complete observation tree and failure handling.
