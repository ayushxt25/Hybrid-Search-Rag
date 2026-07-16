from math import isfinite

from app.embeddings.base import DenseEmbeddingProvider
from app.retrieval.filters import RetrievalFilters
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
        dense_weight: float = 1.5,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        if not isfinite(dense_weight) or not isfinite(sparse_weight):
            raise ValueError("hybrid weights must be finite.")

        if dense_weight <= 0 or sparse_weight <= 0:
            raise ValueError("hybrid weights must be greater than zero.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(
        self,
        request: HybridSearchRequest,
    ) -> HybridSearchResponse:
        """Retrieve dense and sparse candidates, then fuse the rankings."""
        normalized_query = request.query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        filters = RetrievalFilters.from_legacy(
            document_id=request.document_id,
            document_ids=request.document_ids,
            content_types=request.content_types,
        )
        dense_embedding = self.embedding_provider.embed_query(normalized_query)
        sparse_embedding = self.sparse_embedding_provider.embed_query(normalized_query)

        dense_results = self.vector_store.search_dense(
            query_vector=dense_embedding.vector,
            limit=request.candidate_limit,
            filters=filters,
        )
        sparse_results = self.vector_store.search_sparse(
            query_indices=sparse_embedding.indices,
            query_values=sparse_embedding.values,
            limit=request.candidate_limit,
            filters=filters,
        )

        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            weights=[
                self.dense_weight,
                self.sparse_weight,
            ],
            limit=request.limit,
            k=self.rrf_k,
            include_score_diagnostics=request.include_score_diagnostics,
        )

        return HybridSearchResponse(
            query=normalized_query,
            result_count=len(fused_results),
            results=fused_results,
        )
