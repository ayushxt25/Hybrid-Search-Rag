import math
from typing import Self

from pydantic import BaseModel, Field, model_validator


class ChunkEmbedding(BaseModel):
    """Dense embedding generated for one document chunk."""

    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    vector: list[float] = Field(min_length=1)
    dimensions: int = Field(gt=0)


class QueryEmbedding(BaseModel):
    """Dense embedding generated for a search query."""

    query: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    dimensions: int = Field(gt=0)


class SparseEmbeddingValidationMixin(BaseModel):
    """Shared validation for sparse lexical vectors."""

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def validate_sparse_vector(self) -> Self:
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have equal lengths.")

        if any(index < 0 for index in self.indices):
            raise ValueError("indices must be non-negative.")

        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices must be unique.")

        if self.indices != sorted(self.indices):
            raise ValueError("indices must be sorted in ascending order.")

        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("values must contain only finite numbers.")

        if not self.indices:
            raise ValueError("sparse vectors cannot be empty.")

        return self


class ChunkSparseEmbedding(SparseEmbeddingValidationMixin):
    """Sparse lexical embedding generated for one document chunk."""

    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)


class QuerySparseEmbedding(SparseEmbeddingValidationMixin):
    """Sparse lexical embedding generated for a search query."""

    query: str = Field(min_length=1)
