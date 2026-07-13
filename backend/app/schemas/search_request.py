from typing import Self

from pydantic import BaseModel, Field, model_validator

from app.schemas.search import DenseSearchResult


class DenseSearchRequest(BaseModel):
    """Input accepted by the dense-search endpoint."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    document_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )


class DenseSearchResponse(BaseModel):
    """Ranked chunks returned by dense semantic search."""

    query: str
    result_count: int = Field(ge=0)
    results: list[DenseSearchResult]


class SparseSearchRequest(BaseModel):
    """Input accepted by the sparse-search endpoint."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    document_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )


class SparseSearchResponse(BaseModel):
    """Ranked chunks returned by sparse lexical search."""

    query: str
    result_count: int = Field(ge=0)
    results: list[DenseSearchResult]


class HybridSearchRequest(BaseModel):
    """Input accepted by the hybrid-search endpoint."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    document_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> Self:
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be greater than or equal to limit.")

        return self


class HybridSearchResponse(BaseModel):
    """Ranked chunks returned by hybrid dense and sparse search."""

    query: str
    result_count: int = Field(ge=0)
    results: list[DenseSearchResult]
