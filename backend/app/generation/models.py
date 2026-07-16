from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from app.retrieval.filters import RetrievalFilters


class GenerationOutput(BaseModel):
    """Provider output with character-count accounting."""

    text: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    input_characters: int = Field(ge=0)
    output_characters: int = Field(ge=0)
    finish_reason: str | None = None

    @field_validator("text", "model_name")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank.")

        return value

    @field_validator("finish_reason")
    @classmethod
    def validate_finish_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("finish_reason cannot be blank.")

        return value

    @model_validator(mode="after")
    def validate_output_characters(self) -> Self:
        if self.output_characters != len(self.text):
            raise ValueError("output_characters must equal len(text).")

        return self


class GroundedAnswerRequest(BaseModel):
    """Input for grounded answer orchestration."""

    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    document_id: str | None = Field(default=None, min_length=64, max_length=64)
    document_ids: list[str] | None = None
    content_types: list[str] | None = None

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> Self:
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be greater than or equal to limit.")

        RetrievalFilters.from_legacy(
            document_id=self.document_id,
            document_ids=self.document_ids,
            content_types=self.content_types,
        )
        return self


class AnswerCitation(BaseModel):
    """One structured citation source supplied with an answer."""

    source_number: int = Field(gt=0)
    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    file_name: str = Field(min_length=1)
    heading: str | None = None
    page_number: int | None = Field(default=None, gt=0)

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("file_name cannot be blank.")

        return value


class GroundedAnswerResult(BaseModel):
    """Grounded answer plus retrieval/context/generation metadata."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    citations: list[AnswerCitation]
    citation_markers: list[int]
    retrieved_result_count: int = Field(ge=0)
    context_source_count: int = Field(ge=0)
    context_truncated: bool
    insufficient_context: bool
    input_characters: int = Field(ge=0)
    output_characters: int = Field(ge=0)
    finish_reason: str | None = None

    @field_validator("question", "answer", "model_name")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank.")

        return value

    @field_validator("finish_reason")
    @classmethod
    def validate_finish_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("finish_reason cannot be blank.")

        return value

    @field_validator("citation_markers")
    @classmethod
    def validate_citation_markers(cls, value: list[int]) -> list[int]:
        if any(marker <= 0 for marker in value):
            raise ValueError("citation_markers must contain positive values.")

        return value

    @model_validator(mode="after")
    def validate_derived_fields(self) -> Self:
        if self.context_source_count != len(self.citations):
            raise ValueError("context_source_count must equal len(citations).")

        if self.output_characters != len(self.answer):
            raise ValueError("output_characters must equal len(answer).")

        if self.context_source_count == 0 and not self.insufficient_context:
            raise ValueError(
                "insufficient_context must be true when context_source_count is zero."
            )

        if self.insufficient_context and self.citation_markers:
            raise ValueError(
                "citation_markers must be empty when insufficient_context is true."
            )

        return self
