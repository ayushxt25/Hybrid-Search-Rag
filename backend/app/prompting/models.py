from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from app.context.models import AssembledContext


class GroundedPromptRequest(BaseModel):
    """Input for grounded prompt construction."""

    question: str = Field(max_length=2000)
    context: AssembledContext

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question cannot be blank.")

        return value


class GroundedPromptPackage(BaseModel):
    """Rendered prompts plus prompt metadata for grounded generation."""

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_count: int = Field(ge=0)
    context_characters: int = Field(ge=0)
    context_truncated: bool
    insufficient_context: bool
    total_prompt_characters: int = Field(ge=0)

    @field_validator("system_prompt", "user_prompt", "question")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank.")

        return value

    @model_validator(mode="after")
    def validate_derived_fields(self) -> Self:
        if self.total_prompt_characters != len(self.system_prompt) + len(
            self.user_prompt
        ):
            raise ValueError("total_prompt_characters must equal prompt text lengths.")

        if self.source_count == 0 and not self.insufficient_context:
            raise ValueError(
                "insufficient_context must be true when source_count is zero."
            )

        return self
