from abc import ABC, abstractmethod

from app.schemas.document import TextChunk
from app.schemas.embedding import ChunkSparseEmbedding, QuerySparseEmbedding


class SparseEmbeddingProvider(ABC):
    """Interface implemented by sparse lexical embedding providers."""

    @abstractmethod
    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[ChunkSparseEmbedding]:
        """Generate sparse embeddings for document chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> QuerySparseEmbedding:
        """Generate a sparse embedding for a search query."""
