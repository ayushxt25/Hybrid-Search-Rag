from pydantic import BaseModel, Field, model_validator


class RateLimitDecision(BaseModel):
    allowed: bool
    limit: int = Field(gt=0)
    remaining: int = Field(ge=0)
    reset_after_seconds: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_remaining_within_limit(self) -> "RateLimitDecision":
        if self.remaining > self.limit:
            raise ValueError("remaining must be less than or equal to limit.")

        return self
