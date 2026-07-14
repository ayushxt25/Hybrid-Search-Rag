from math import isfinite
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ContextSource(BaseModel):
    """One cited source selected for generated-answer context."""

    source_number: int = Field(gt=0)
    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    file_name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float
    chunk_index: int = Field(ge=0)
    section_index: int | None = Field(default=None, ge=0)
    heading: str | None = None
    page_number: int | None = Field(default=None, gt=0)

    @field_validator("file_name", "text")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank.")

        return value

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("score must be finite.")

        return value


class AssembledContext(BaseModel):
    """Rendered context text plus structured citation sources."""

    context_text: str
    sources: list[ContextSource]
    source_count: int
    total_characters: int
    truncated: bool
    omitted_result_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_derived_counts(self) -> Self:
        if self.source_count != len(self.sources):
            raise ValueError("source_count must equal len(sources).")

        if self.total_characters != len(self.context_text):
            raise ValueError("total_characters must equal len(context_text).")

        return self
