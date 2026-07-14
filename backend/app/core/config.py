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
    log_level: str = "INFO"
    observability_enabled: bool = True

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
    grounded_answer_rate_limit_enabled: bool = True
    grounded_answer_rate_limit_requests: int = Field(default=10, gt=0)
    grounded_answer_rate_limit_window_seconds: int = Field(default=60, gt=0)
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_generation_model: str = "gpt-4.1-mini"
    openai_generation_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_generation_max_retries: int = Field(default=2, ge=0)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized_value = value.upper()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized_value not in allowed_levels:
            raise ValueError(
                "log_level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )

        return normalized_value

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def normalize_blank_openai_base_url(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None

        return value

    @field_validator("openai_generation_timeout_seconds", mode="before")
    @classmethod
    def validate_openai_timeout(cls, value: object) -> object:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value

        if not isfinite(numeric_value):
            raise ValueError("openai timeout must be finite.")

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
