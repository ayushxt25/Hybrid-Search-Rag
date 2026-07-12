from pydantic import BaseModel, Field


class ChunkEmbedding(BaseModel):
    """Dense embedding generated for one document chunk."""

    chunk_id: str = Field(min_length=64, max_length=64)
    document_id: str = Field(min_length=64, max_length=64)
    vector: list[float] = Field(min_length=1)
    dimensions: int = Field(gt=0)


class QueryEmbedding(BaseModel):
    """Dense embedding generated for a search query."""

    query: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    dimensions: int = Field(gt=0)
