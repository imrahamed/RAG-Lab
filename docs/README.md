# RAG Lab documentation

**Understand the system. Run each stage. Inspect the evidence.**

RAG Lab uses remote PostgreSQL and OpenSearch, local Ollama, and optional hosted Langfuse. The guides follow the same path as the code: ingest knowledge, retrieve evidence, generate an answer, and inspect its trace.

## Choose your starting point

| Your goal | Start here | What you will leave with |
| --- | --- | --- |
| Run the project | [Getting started](getting-started.md) | Healthy services and your first grounded answer |
| Learn by executing code | [Jupyter walkthrough](../notebooks/rag_walkthrough.ipynb) · [Notebook guide](../notebooks/README.md) | An executable tour, including offline experiments |
| Understand the service boundaries | [Architecture](architecture.md) | A map of the ingestion and question paths |
| Read the implementation | [Implementation walkthrough](implementation.md) | The purpose and tradeoffs of each module |
| Follow a learning sequence | [Hands-on lessons](lessons.md) | Eight exercises with expected outcomes |
| Configure a deployment | [Configuration](configuration.md) | Environment settings, model identity, and TLS guidance |
| Call the application | [API reference](api.md) | Request examples and response contracts |
| Inspect latency and model calls | [Langfuse tracing](observability.md) | Trace trees, token usage, and delivery checks |
| Diagnose or rebuild | [Operations](operations.md) | Initialization, recovery, and troubleshooting commands |

## A useful learning order

1. **Connect:** complete setup and run `rag doctor`.
2. **Inspect:** open the notebook and load the fictional sample corpus.
3. **Retrieve:** compare keyword, semantic, and hybrid results before generating answers.
4. **Ground:** map the answer's numbered citations to the supplied excerpts.
5. **Adapt:** inspect the bounded grade → rewrite → retrieve workflow.
6. **Measure:** compare retrieval metrics and examine a Langfuse trace.

The notebook's default offline run does not need credentials or a model server. Live cells use the same application services as the CLI and API. All example company policies belong to the fictional Northstar corpus.

---

[Project overview](../README.md) · [Source code](../src/rag_lab) · [Notebook](../notebooks/rag_walkthrough.ipynb)
