import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    postgres_dsn: SecretStr
    opensearch_url: AnyHttpUrl
    opensearch_username: str | None = None
    opensearch_password: SecretStr | None = None
    opensearch_index: str = Field(default="rag-lab-v1", pattern=r"^[a-z][a-z0-9_-]*$")
    opensearch_ca_cert: Path | None = None
    ollama_url: AnyHttpUrl = "http://127.0.0.1:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_context_tokens: int = Field(default=8192, ge=2048)
    embed_document_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    embedding_dimensions: int = Field(default=768, ge=1)
    chunk_size: int = Field(default=1400, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    embed_batch_size: int = Field(default=16, ge=1, le=128)
    retrieval_candidates: int = Field(default=30, ge=10, le=1000)
    max_context_chars: int = Field(default=12000, ge=1000)
    request_timeout_seconds: float = Field(default=180, gt=0)
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: AnyHttpUrl = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    langfuse_capture_content: bool = False
    langfuse_timeout_seconds: int = Field(default=5, ge=1, le=60)

    @model_validator(mode="after")
    def validate_relationships(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if bool(self.opensearch_username) != bool(self.opensearch_password):
            raise ValueError("Set both OpenSearch username and password, or neither")
        if self.opensearch_url.username or self.opensearch_url.password:
            raise ValueError("Use OPENSEARCH_USERNAME/PASSWORD instead of URL credentials")
        if self.langfuse_enabled and not (self.langfuse_public_key and self.langfuse_secret_key):
            raise ValueError("Enabled Langfuse tracing requires both project API keys")
        if self.langfuse_base_url.username or self.langfuse_base_url.password:
            raise ValueError("Use Langfuse project API keys instead of URL credentials")
        return self

    @property
    def embedding_profile(self) -> dict:
        # Changing a model with the same dimension still changes the vector space.
        return {
            "model": self.ollama_embed_model,
            "dimensions": self.embedding_dimensions,
            "document_prefix": self.embed_document_prefix,
            "query_prefix": self.embed_query_prefix,
        }

    @property
    def pipeline_version(self) -> str:
        value = {
            **self.embedding_profile,
            "size": self.chunk_size,
            "overlap": self.chunk_overlap,
            "chunker": "characters-v1",
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


SearchMode = Literal["keyword", "semantic", "hybrid"]
