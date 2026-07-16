from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.documents.service import validate_document_id

MAX_DOCUMENT_FILTERS = 20
MAX_CONTENT_TYPE_FILTERS = 10
SUPPORTED_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _normalize_unique(values: list[str], *, lowercase: bool = False) -> list[str]:
    normalized_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if lowercase:
            normalized = normalized.lower()
        if not normalized:
            raise ValueError("filter values cannot be blank.")
        if normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return normalized_values


class RetrievalFilters(BaseModel):
    document_ids: list[str] = Field(
        default_factory=list,
    )
    content_types: list[str] = Field(
        default_factory=list,
    )

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, value: list[str]) -> list[str]:
        return [validate_document_id(item) for item in _normalize_unique(value)]

    @field_validator("content_types")
    @classmethod
    def validate_content_types(cls, value: list[str]) -> list[str]:
        normalized_values = _normalize_unique(value, lowercase=True)
        unsupported = [
            item for item in normalized_values if item not in SUPPORTED_CONTENT_TYPES
        ]
        if unsupported:
            raise ValueError("unsupported content_type filter.")
        return normalized_values

    @model_validator(mode="after")
    def validate_counts_after_deduplication(self) -> Self:
        if len(self.document_ids) > MAX_DOCUMENT_FILTERS:
            raise ValueError(
                f"document_ids cannot contain more than {MAX_DOCUMENT_FILTERS} values."
            )
        if len(self.content_types) > MAX_CONTENT_TYPE_FILTERS:
            raise ValueError(
                "content_types cannot contain more than "
                f"{MAX_CONTENT_TYPE_FILTERS} values."
            )
        return self

    @classmethod
    def from_legacy(
        cls,
        *,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        content_types: list[str] | None = None,
    ) -> "RetrievalFilters":
        merged_document_ids = []
        if document_id is not None:
            normalized_document_id = document_id.strip()
            if not normalized_document_id:
                raise ValueError("document_id cannot be empty")
            merged_document_ids.append(normalized_document_id)
        if document_ids is not None:
            merged_document_ids.extend(document_ids)
        return cls(
            document_ids=merged_document_ids,
            content_types=[] if content_types is None else content_types,
        )

    model_config = ConfigDict(frozen=True)
