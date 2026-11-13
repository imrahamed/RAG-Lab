import pytest
from pydantic import ValidationError

from rag_lab.config import Settings


def test_overlap_and_partial_credentials_are_rejected(settings):
    values = settings.model_dump()
    for changes in (
        {"chunk_overlap": 100},
        {"opensearch_username": "user"},
        {"opensearch_url": "https://user:secret@search.example.com"},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{**values, **changes})


def test_embedding_profile_tracks_model_and_prefixes(settings):
    changed = settings.model_copy(update={"ollama_embed_model": "another-model"})
    assert settings.embedding_profile != changed.embedding_profile
    assert settings.pipeline_version != changed.pipeline_version
    changed = settings.model_copy(update={"embed_query_prefix": ""})
    assert settings.embedding_profile != changed.embedding_profile
    assert settings.pipeline_version != changed.pipeline_version


def test_chunk_change_invalidates_revision_but_not_vector_space(settings):
    changed = settings.model_copy(update={"chunk_size": 200})
    assert changed.embedding_profile == settings.embedding_profile
    assert changed.pipeline_version != settings.pipeline_version
