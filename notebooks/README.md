# The RAG Lab notebook

**Read a concept, execute the code, inspect what changed.**

[Open the notebook](rag_walkthrough.ipynb) · [Read the HTML preview](rag_walkthrough.html) · [Documentation](../docs/README.md)

The notebook follows this project's actual modules rather than reimplementing a second RAG application. Its offline experiments run without service credentials. Its live sections use the same `RagService` as the CLI and API.

## Launch

Run from the project root:

```powershell
uv sync --extra dev --extra notebook
uv run --extra notebook jupyter lab notebooks/rag_walkthrough.ipynb
```

Use the Python kernel from this project's `.venv`. When launched with the command above, Jupyter runs in the project environment. In an editor such as VS Code, select `.venv/Scripts/python.exe` explicitly. Run cells from top to bottom; restart the kernel after changing `.env`.

## Choose an execution mode

The controls cell starts with all service operations off:

| Switch | Default | Effect |
| --- | --- | --- |
| `RUN_LIVE` | `False` | Connect to your configured services; enable actual retrieval and generation |
| `INITIALIZE_REMOTE` | `False` | With live mode on, run application initialization |
| `INGEST_SAMPLE` | `False` | With live mode on, write the three sample documents to your configured stores |
| `RUN_EVALUATION` | `False` | With live mode on, run the three-query comparison across all retrieval modes |

**First pass:** leave defaults unchanged and choose Run All. You can inspect the sample corpus, change chunking, calculate RRF scores, and build an evidence context. No `.env` is loaded and no network clients are created.

**Live pass:** configure `.env` as described in [getting started](../docs/getting-started.md). Set `RUN_LIVE=True`. If your resources are not initialized, also set `INITIALIZE_REMOTE=True`. If the sample corpus is not yet stored, set `INGEST_SAMPLE=True`. Enable `RUN_EVALUATION` when you want the extra retrieval calls. These flags write only the application's configured tables/index and the fictional sample corpus; they do not delete stored documents or indexes.

Live generation can be slow when a local model is loading. If a cell fails, inspect the short diagnostic, fix the dependency, and rerun from the relevant setup cell. Run the final cleanup cell when finished so buffered Langfuse observations are drained. Close the existing service before rerunning its creation cell.

## The notebook journey

| Section | Experiment |
| --- | --- |
| 01–02 | Locate the project, choose a mode, and check configured services |
| 03–04 | Read the sample corpus and inspect overlapping chunks |
| 05–06 | Generate real local embeddings and ingest knowledge when live mode is enabled |
| 07–08 | Compare retrieval modes and calculate an illustrative RRF ranking |
| 09–10 | Build bounded evidence and inspect an actual cited answer |
| 11 | Observe the grade/rewrite/retrieve path on an unsupported question |
| 12–13 | Compare retrieval metrics and inspect trace IDs |
| 14 | Close clients and choose the next exercise |

## Validate and refresh the preview

```powershell
uv run --extra notebook python scripts/check_notebook.py
```

This validates the notebook schema and code, executes the offline path in an isolated temporary kernel configuration, and refreshes `rag_walkthrough.html`. The source notebook remains output-free for clean diffs and to avoid saving future live prompts or service data. The HTML preview contains only offline sample outputs.

The validator deliberately prevents network connections in the execution kernel. Live service execution is a separate user-selected run, using the flags above. The preview's results are offline demonstrations, not a claim that a model generated an answer or achieved a retrieval score.

## Documentation images

The notebook includes the project cover and illustrations for architecture, chunking, hybrid retrieval, evidence, adaptive search, and tracing. The PNG files live in [`docs/assets`](../docs/assets/) and are shared with the documentation and HTML preview. Captions, tables, and executable explanations accompany the images.
