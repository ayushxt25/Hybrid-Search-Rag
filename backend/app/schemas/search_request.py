from pydantic import BaseModel, Field

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
