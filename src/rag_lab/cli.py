import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from rag_lab.config import Settings
from rag_lab.models import AskRequest, SearchRequest
from rag_lab.pipeline import RagService


def main():
    parser = argparse.ArgumentParser(description="RAG Lab: learn each pipeline step")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create application tables and a versioned search index")
    commands.add_parser("doctor", help="Check all three services and installed models")
    ingest = commands.add_parser("ingest", help="Ingest one file or a directory recursively")
    ingest.add_argument("path", type=Path)
    ingest.add_argument(
        "--force", action="store_true", help="Rebuild even when content is unchanged"
    )
    arxiv = commands.add_parser("arxiv", help="Fetch and ingest a single arXiv PDF")
    arxiv.add_argument("paper_id")
    commands.add_parser("reindex", help="Rebuild search chunks from canonical PostgreSQL content")
    commands.add_parser("documents", help="List the most recent 100 documents")
    for name in ("search", "ask"):
        command = commands.add_parser(name)
        command.add_argument("query")
        command.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid")
        command.add_argument("--top-k", type=int, default=5)
        if name == "ask":
            command.add_argument(
                "--agentic",
                action="store_true",
                help="Grade excerpts and rewrite/retrieve at most once",
            )
    args = parser.parse_args()
    service = None
    try:
        service = RagService(Settings())
        match args.command:
            case "init":
                service.initialize()
                result = {"status": "initialized"}
            case "doctor":
                result = service.health(check_tracing=True)
            case "ingest":
                if not args.path.exists():
                    raise ValueError("Input path does not exist")
                files = (
                    sorted(
                        p
                        for p in args.path.rglob("*")
                        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}
                    )
                    if args.path.is_dir()
                    else [args.path]
                )
                if not files:
                    raise ValueError("No supported documents found")
                # Emit per-file progress immediately; a retry skips already committed files.
                for path in files:
                    print(json.dumps(service.ingest_file(path, force=args.force)))
                return
            case "arxiv":
                result = service.ingest_arxiv(args.paper_id)
            case "reindex":
                for document_id in service.db.document_ids():
                    print(json.dumps(service.reindex(document_id)))
                return
            case "documents":
                result = service.db.documents()
            case "search":
                result = [
                    hit.model_dump()
                    for hit in service.search(
                        SearchRequest(query=args.query, mode=args.mode, top_k=args.top_k)
                    )
                ]
            case "ask":
                result = service.ask(
                    AskRequest(
                        query=args.query, mode=args.mode, top_k=args.top_k, agentic=args.agentic
                    )
                ).model_dump()
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        if args.command == "doctor" and result["status"] != "ok":
            sys.exit(1)
    except ValidationError as error:
        # Do not echo raw input values (which may contain credentials).
        fields = [".".join(map(str, item["loc"])) or "settings" for item in error.errors()]
        print("Invalid configuration/input: " + ", ".join(fields), file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        message = (
            str(error)
            if isinstance(error, (ValueError, FileNotFoundError))
            else (
                f"{type(error).__name__}: operation failed. Run `rag doctor`; check service logs."
            )
        )
        print(message, file=sys.stderr)
        sys.exit(1)
    finally:
        if service:
            service.close()


if __name__ == "__main__":
    main()
