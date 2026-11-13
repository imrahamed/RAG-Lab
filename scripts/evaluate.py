"""Small retrieval-only exercise. Run after ingesting data/sample."""

import argparse
import json
from pathlib import Path

from rag_lab.config import Settings
from rag_lab.models import SearchRequest
from rag_lab.pipeline import RagService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "data/evaluation.json"
    )
    args = parser.parse_args()
    examples = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not examples:
        parser.error("The evaluation dataset must contain at least one question")
    service = RagService(Settings())
    rows = []
    try:
        for example in examples:
            hits = service.search(SearchRequest(query=example["query"], mode=args.mode, top_k=5))
            # Evaluate document ranks, deduplicating multiple chunks from the same source.
            titles = list(dict.fromkeys(hit.title for hit in hits))
            rank = (
                titles.index(example["expected_title"]) + 1
                if example["expected_title"] in titles
                else None
            )
            rows.append({"query": example["query"], "rank": rank})
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "queries": len(rows),
                    "document_recall_at_5": sum(row["rank"] is not None for row in rows)
                    / len(rows),
                    "mean_reciprocal_rank": sum(
                        1 / row["rank"] if row["rank"] else 0 for row in rows
                    )
                    / len(rows),
                    "results": rows,
                },
                indent=2,
            )
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
