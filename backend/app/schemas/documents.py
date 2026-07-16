from datetime import datetime

from pydantic import BaseModel, Field


class IndexedDocumentSummary(BaseModel):
    document_id: str = Field(min_length=64, max_length=64)
    filename: str | None = None
    content_type: str | None = None
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    chunk_count: int = Field(ge=0)
    indexed_at: datetime | None = None


class IndexedDocumentListResponse(BaseModel):
    documents: list[IndexedDocumentSummary]
    next_cursor: str | None = None


class IndexedDocumentDetail(IndexedDocumentSummary):
    chunk_indices: list[int]
    page_numbers: list[int]
    headings: list[str]


class DocumentDeletionResponse(BaseModel):
    document_id: str = Field(min_length=64, max_length=64)
    deleted_chunks: int = Field(ge=0)
    deleted: bool
