from pydantic import BaseModel, Field


class IndexedDocumentResult(BaseModel):
    """Summary returned after a document is indexed successfully."""

    document_id: str = Field(min_length=64, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64)

    file_name: str = Field(min_length=1)
    file_extension: str = Field(min_length=1)

    chunk_count: int = Field(gt=0)
    indexed_points: int = Field(gt=0)
