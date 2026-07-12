from sentence_transformers import SentenceTransformer

from app.embeddings.base import DenseEmbeddingProvider
from app.schemas.document import TextChunk
from app.schemas.embedding import ChunkEmbedding, QueryEmbedding


class SentenceTransformerEmbeddingProvider(DenseEmbeddingProvider):
    """Generate normalized dense vectors using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

        dimensions = self._model.get_embedding_dimension()

        if dimensions is None or dimensions <= 0:
            raise RuntimeError("Embedding model did not report a valid dimension.")

        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Return the model embedding dimension."""
        return self._dimensions

    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[ChunkEmbedding]:
        """Generate one normalized vector for every supplied chunk."""
        if not chunks:
            return []

        chunk_texts = [chunk.text for chunk in chunks]

        vectors = self._model.encode_document(
            chunk_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding model returned an unexpected vector count.")

        results: list[ChunkEmbedding] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            vector_values = vector.tolist()

            if len(vector_values) != self.dimensions:
                raise RuntimeError("Embedding vector has an unexpected dimension.")

            results.append(
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    vector=vector_values,
                    dimensions=self.dimensions,
                )
            )

        return results

    def embed_query(self, query: str) -> QueryEmbedding:
        """Generate a normalized vector for a non-empty query."""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        vector = self._model.encode_query(
            normalized_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vector_values = vector.tolist()

        if len(vector_values) != self.dimensions:
            raise RuntimeError("Query embedding has an unexpected dimension.")

        return QueryEmbedding(
            query=normalized_query,
            vector=vector_values,
            dimensions=self.dimensions,
        )
