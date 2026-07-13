from app.embeddings.base import DenseEmbeddingProvider
from app.retrieval.rrf import reciprocal_rank_fusion
from app.schemas.search_request import (
    HybridSearchRequest,
    HybridSearchResponse,
)
from app.sparse.base import SparseEmbeddingProvider
from app.vectorstore.qdrant import QdrantVectorStore


class HybridSearchService:
    """Coordinate dense and sparse retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        embedding_provider: DenseEmbeddingProvider,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_store: QdrantVectorStore,
        rrf_k: int = 60,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def search(
        self,
        request: HybridSearchRequest,
    ) -> HybridSearchResponse:
        """Retrieve dense and sparse candidates, then fuse the rankings."""
        normalized_query = request.query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        dense_embedding = self.embedding_provider.embed_query(normalized_query)
        sparse_embedding = self.sparse_embedding_provider.embed_query(normalized_query)

        dense_results = self.vector_store.search_dense(
            query_vector=dense_embedding.vector,
            limit=request.candidate_limit,
            document_id=request.document_id,
        )
        sparse_results = self.vector_store.search_sparse(
            query_indices=sparse_embedding.indices,
            query_values=sparse_embedding.values,
            limit=request.candidate_limit,
            document_id=request.document_id,
        )

        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            limit=request.limit,
            k=self.rrf_k,
        )

        return HybridSearchResponse(
            query=normalized_query,
            result_count=len(fused_results),
            results=fused_results,
        )
