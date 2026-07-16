from math import isfinite
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)


class BranchScoreDiagnostic(BaseModel):
    """Safe score details for one retrieval branch."""

    raw_score: float | None
    rank: int | None = Field(default=None, ge=1)
    weight: float = Field(gt=0)
    rrf_contribution: float = Field(ge=0)

    @field_validator("raw_score", "weight", "rrf_contribution")
    @classmethod
    def validate_finite_number(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("numeric diagnostics must be finite.")
        return value

    model_config = ConfigDict(frozen=True)


class RetrievalScoreDiagnostic(BaseModel):
    """Safe retrieval score diagnostics for one returned chunk."""

    dense: BranchScoreDiagnostic
    sparse: BranchScoreDiagnostic
    fused_score: float = Field(ge=0)
    fused_rank: int = Field(ge=1)

    @field_validator("fused_score")
    @classmethod
    def validate_finite_fused_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("numeric diagnostics must be finite.")
        return value

    @model_validator(mode="after")
    def validate_fused_score_matches_contributions(self) -> Self:
        expected_score = self.dense.rrf_contribution + self.sparse.rrf_contribution
        if self.fused_score < 0 or not isfinite(expected_score):
            raise ValueError("numeric diagnostics must be finite.")
        return self

    model_config = ConfigDict(frozen=True)


class DenseSearchResult(BaseModel):
    """One document chunk returned by dense vector search."""

    point_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    score: float
    score_diagnostics: RetrievalScoreDiagnostic | None = None

    file_name: str = Field(min_length=1)
    file_extension: str = Field(min_length=1)

    chunk_index: int = Field(ge=0)
    section_index: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    heading: str | None = None

    text: str = Field(min_length=1)
    start_word: int = Field(ge=0)
    end_word: int = Field(gt=0)
    word_count: int = Field(gt=0)

    @model_serializer(mode="wrap")
    def serialize_result(self, handler: SerializerFunctionWrapHandler) -> dict:
        data = handler(self)
        if self.score_diagnostics is None:
            data.pop("score_diagnostics", None)
        return data
