from functools import lru_cache
from math import isfinite

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Hybrid Search RAG"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "internal_document_chunks"
    qdrant_hybrid_collection_name: str = "internal_document_chunks_hybrid"
    dense_embedding_dimensions: int = Field(default=384, gt=0)
    hybrid_dense_weight: float = 1.5
    hybrid_sparse_weight: float = 1.0
    hybrid_rrf_k: int = Field(default=60, gt=0)
    context_max_characters: int = Field(default=12000, gt=0)
    context_max_sources: int = Field(default=8, gt=0)
    context_include_metadata_headers: bool = True
    prompt_max_question_characters: int = Field(default=2000, gt=0)
    prompt_require_citations: bool = True
    prompt_allow_general_knowledge: bool = False
    generation_require_answer_citations: bool = True
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_generation_model: str = "gpt-4.1-mini"

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def normalize_blank_openai_base_url(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None

        return value

    @field_validator("hybrid_dense_weight", "hybrid_sparse_weight")
    @classmethod
    def validate_hybrid_weight(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("hybrid weights must be finite.")

        if value <= 0:
            raise ValueError("hybrid weights must be greater than zero.")

        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
