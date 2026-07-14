from math import isfinite

from pydantic import BaseModel, field_validator


class GroundedAnswerTimings(BaseModel):
    retrieval_ms: float
    context_assembly_ms: float
    prompt_construction_ms: float
    generation_ms: float
    total_ms: float

    @field_validator("*")
    @classmethod
    def validate_non_negative_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("timing values must be finite.")
        if value < 0:
            raise ValueError("timing values must be non-negative.")
        return value
