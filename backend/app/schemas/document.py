from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """A logical source section extracted from a document."""

    section_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    heading: str | None = None


class LoadedDocument(BaseModel):
    """Text and metadata extracted from an uploaded document."""

    document_id: str = Field(min_length=64, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64)
    file_name: str = Field(min_length=1)
    file_extension: str = Field(min_length=1)
    source_path: Path
    content: str = Field(min_length=1)
    character_count: int = Field(gt=0)
    word_count: int = Field(gt=0)
    sections: list[DocumentSection] = Field(min_length=1)


class TextChunk(BaseModel):
    """A searchable passage produced from an internal document."""

    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    chunk_index: int = Field(ge=0)
    section_index: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    heading: str | None = None
    text: str = Field(min_length=1)
    start_word: int = Field(ge=0)
    end_word: int = Field(gt=0)
    word_count: int = Field(gt=0)


class IngestedDocument(BaseModel):
    """A loaded document together with its generated chunks."""

    document: LoadedDocument
    chunks: list[TextChunk] = Field(min_length=1)
    chunk_count: int = Field(gt=0)


class DocumentChunkResponse(BaseModel):
    """Chunk information returned by the ingestion API."""

    chunk_id: str
    document_id: str
    chunk_index: int
    section_index: int
    page_number: int | None
    heading: str | None
    text: str
    start_word: int
    end_word: int
    word_count: int


class DocumentIngestionResponse(BaseModel):
    """Successful document-ingestion API response."""

    status: Literal["processed"]
    document_id: str
    content_hash: str
    file_name: str
    file_extension: str
    character_count: int
    word_count: int
    chunk_count: int
    chunks: list[DocumentChunkResponse]
