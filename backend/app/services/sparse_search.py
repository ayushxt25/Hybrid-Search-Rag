from app.retrieval.filters import RetrievalFilters
from app.schemas.search_request import (
    SparseSearchRequest,
    SparseSearchResponse,
)
from app.sparse.base import SparseEmbeddingProvider
from app.vectorstore.qdrant import QdrantVectorStore


class SparseSearchService:
    """Coordinate sparse query encoding and Qdrant sparse retrieval."""

    def __init__(
        self,
        *,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_store = vector_store

    def search(
        self,
        request: SparseSearchRequest,
    ) -> SparseSearchResponse:
        """Encode a lexical query and retrieve matching document chunks."""
        normalized_query = request.query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        filters = RetrievalFilters.from_legacy(
            document_id=request.document_id,
            document_ids=request.document_ids,
            content_types=request.content_types,
        )
        query_embedding = self.sparse_embedding_provider.embed_query(normalized_query)

        results = self.vector_store.search_sparse(
            query_indices=query_embedding.indices,
            query_values=query_embedding.values,
            limit=request.limit,
            filters=filters,
        )

        return SparseSearchResponse(
            query=normalized_query,
            result_count=len(results),
            results=results,
        )
