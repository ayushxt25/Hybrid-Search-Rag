from pathlib import Path

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
