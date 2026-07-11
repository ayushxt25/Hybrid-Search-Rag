from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """Text and metadata extracted from an uploaded document."""

    file_name: str = Field(min_length=1)
    file_extension: str = Field(min_length=1)
    source_path: Path
    content: str = Field(min_length=1)
    character_count: int = Field(gt=0)
    word_count: int = Field(gt=0)


class TextChunk(BaseModel):
    """A searchable passage produced from an internal document."""

    chunk_index: int = Field(ge=0)
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

    chunk_index: int
    text: str
    start_word: int
    end_word: int
    word_count: int


class DocumentIngestionResponse(BaseModel):
    """Successful document-ingestion API response."""

    status: Literal["processed"]
    file_name: str
    file_extension: str
    character_count: int
    word_count: int
    chunk_count: int
    chunks: list[DocumentChunkResponse]
