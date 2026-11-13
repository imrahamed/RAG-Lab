"""Deterministic overlapping character windows with reproducible positions."""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer a word boundary in the final quarter of a full window.
        if end < len(text):
            boundary = text.rfind(" ", start + size * 3 // 4, end)
            if boundary > start + overlap:
                end = boundary
        value = text[start:end].strip()
        if value:
            chunks.append(value)
        if end == len(text):
            break
        start = end - overlap
    return chunks
