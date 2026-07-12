from app.embeddings.base import DenseEmbeddingProvider
from app.schemas.search_request import (
    DenseSearchRequest,
    DenseSearchResponse,
)
from app.vectorstore.qdrant import QdrantVectorStore


class DenseSearchService:
    """Coordinate query embedding and Qdrant dense retrieval."""

    def __init__(
        self,
        *,
        embedding_provider: DenseEmbeddingProvider,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def search(
        self,
        request: DenseSearchRequest,
    ) -> DenseSearchResponse:
        """Embed a query and retrieve the nearest document chunks."""
        normalized_query = request.query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        query_embedding = self.embedding_provider.embed_query(normalized_query)

        if query_embedding.dimensions != self.vector_store.vector_dimensions:
            raise RuntimeError(
                "Query embedding dimensions do not match the vector store."
            )

        results = self.vector_store.search_dense(
            query_vector=query_embedding.vector,
            limit=request.limit,
            score_threshold=request.score_threshold,
            document_id=request.document_id,
        )

        return DenseSearchResponse(
            query=normalized_query,
            result_count=len(results),
            results=results,
        )
