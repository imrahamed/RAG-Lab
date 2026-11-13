import pytest

from rag_lab.chunking import chunk_text


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (-1, 0)])
def test_invalid_windows_fail(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("document", size, overlap)


@pytest.mark.parametrize("size,overlap", [(100, 0), (100, 20), (100, 99)])
def test_windows_cover_the_entire_input_without_gaps(size, overlap):
    text = "0123456789" * 127
    chunks = chunk_text(text, size, overlap)
    reconstructed = chunks[0] + "".join(chunk[overlap:] for chunk in chunks[1:])
    assert reconstructed == text
    assert all(0 < len(chunk) <= size for chunk in chunks)


def test_empty_and_normalized_text():
    assert chunk_text(" \n\t", 100, 20) == []
    assert chunk_text("  first\r\nsecond\rthird  ", 100, 20) == ["first\nsecond\nthird"]


def test_word_boundaries_do_not_produce_empty_chunks_or_infinite_loops():
    text = "a word " * 100
    chunks = chunk_text(text, 100, 99)
    assert chunks and all(chunk.strip() for chunk in chunks)
    assert len(chunks) < len(text)
