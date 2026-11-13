import pytest

from rag_lab.config import Settings
from rag_lab.models import Hit


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        postgres_dsn="postgresql://unused:secret@localhost/test",
        opensearch_url="https://search.example.com",
        embedding_dimensions=3,
        chunk_size=100,
        chunk_overlap=20,
    )


@pytest.fixture
def hit():
    return Hit(
        chunk_id="chunk-1",
        document_id="doc-1",
        title="backup-policy",
        source="file:///backup-policy.md",
        position=0,
        text="Daily backups are retained for 30 days.",
        score=0.03,
    )
