from pydantic import BaseModel, Field


class DenseSearchResult(BaseModel):
    """One document chunk returned by dense vector search."""

    point_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    score: float

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
