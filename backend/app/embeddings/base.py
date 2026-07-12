from abc import ABC, abstractmethod

from app.schemas.document import TextChunk
from app.schemas.embedding import ChunkEmbedding, QueryEmbedding


class DenseEmbeddingProvider(ABC):
    """Interface implemented by dense embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the number of dimensions produced by this provider."""

    @abstractmethod
    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[ChunkEmbedding]:
        """Generate embeddings for document chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> QueryEmbedding:
        """Generate an embedding for a search query."""
