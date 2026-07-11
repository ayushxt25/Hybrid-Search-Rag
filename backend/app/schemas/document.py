from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A searchable passage produced from an internal document."""

    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_word: int = Field(ge=0)
    end_word: int = Field(gt=0)
    word_count: int = Field(gt=0)