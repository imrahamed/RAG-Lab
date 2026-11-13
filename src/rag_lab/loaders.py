import io
import re
from pathlib import Path

import httpx
from pypdf import PdfReader


def pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(
        f"[Page {i}]\n{text}"
        for i, page in enumerate(reader.pages, 1)
        if (text := page.extract_text() or "").strip()
    )


def load_file(path: Path) -> tuple[str, str, str]:
    path = path.resolve(strict=True)
    if path.suffix.lower() == ".pdf":
        text = pdf_text(path.read_bytes())
    elif path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
    else:
        raise ValueError("Supported formats: .txt, .md, .pdf")
    if not text.strip():
        raise ValueError("No text extracted. Scanned PDFs need OCR before ingestion.")
    return path.stem, path.as_uri(), text


def load_arxiv(paper_id: str) -> tuple[str, str, str]:
    # An explicit paper ID avoids introducing a crawler or scheduler into the lesson.
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", paper_id):
        raise ValueError("Use an arXiv ID such as 2005.11401, not an arbitrary URL")
    source = f"https://arxiv.org/abs/{paper_id}"
    response = httpx.get(f"https://arxiv.org/pdf/{paper_id}", timeout=120, follow_redirects=True)
    response.raise_for_status()
    text = pdf_text(response.content)
    if not text.strip():
        raise ValueError("No text extracted from the arXiv PDF")
    return f"arXiv {paper_id}", source, text
