from unittest.mock import Mock, patch

import pytest

from app.embeddings.base import DenseEmbeddingProvider
from app.schemas.embedding import QueryEmbedding, QuerySparseEmbedding
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import HybridSearchRequest
from app.services.hybrid_search import HybridSearchService
from app.sparse.base import SparseEmbeddingProvider
from app.vectorstore.qdrant import QdrantVectorStore

DOCUMENT_ID = "a" * 64
CHUNK_ID = "b" * 64


def create_result(
    chunk_id: str = CHUNK_ID,
) -> DenseSearchResult:
    return DenseSearchResult(
        point_id="123e4567-e89b-12d3-a456-426614174000",
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        score=0.85,
        file_name="policy.txt",
        file_extension=".txt",
        chunk_index=0,
        section_index=0,
        page_number=None,
        heading=None,
        text="Employees may work remotely.",
        start_word=0,
        end_word=5,
        word_count=5,
    )


def create_service(
    *,
    dense_weight: float = 1.5,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
) -> tuple[HybridSearchService, Mock, Mock, Mock]:
    embedding_provider = Mock(spec=DenseEmbeddingProvider)
    sparse_embedding_provider = Mock(spec=SparseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    embedding_provider.embed_query.return_value = QueryEmbedding(
        query="remote work policy",
        vector=[0.0] * 384,
        dimensions=384,
    )
    sparse_embedding_provider.embed_query.return_value = QuerySparseEmbedding(
        query="remote work policy",
        indices=[3, 9],
        values=[1.0, 2.0],
    )
    vector_store.search_dense.return_value = [create_result("b" * 64)]
    vector_store.search_sparse.return_value = [create_result("c" * 64)]

    service = HybridSearchService(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
    )

    return service, embedding_provider, sparse_embedding_provider, vector_store


def test_search_retrieves_candidates_and_fuses_results() -> None:
    service, embedding_provider, sparse_embedding_provider, vector_store = (
        create_service()
    )
    fused_result = create_result("d" * 64)

    with patch(
        "app.services.hybrid_search.reciprocal_rank_fusion",
        return_value=[fused_result],
    ) as fusion:
        response = service.search(
            HybridSearchRequest(
                query="  remote work policy  ",
                limit=3,
                candidate_limit=10,
                document_id=DOCUMENT_ID,
            )
        )

    assert response.query == "remote work policy"
    assert response.result_count == 1
    assert response.results == [fused_result]
    embedding_provider.embed_query.assert_called_once_with("remote work policy")
    sparse_embedding_provider.embed_query.assert_called_once_with("remote work policy")
    vector_store.search_dense.assert_called_once_with(
        query_vector=[0.0] * 384,
        limit=10,
        document_id=DOCUMENT_ID,
    )
    vector_store.search_sparse.assert_called_once_with(
        query_indices=[3, 9],
        query_values=[1.0, 2.0],
        limit=10,
        document_id=DOCUMENT_ID,
    )
    fusion.assert_called_once_with(
        [
            vector_store.search_dense.return_value,
            vector_store.search_sparse.return_value,
        ],
        weights=[1.5, 1.0],
        limit=3,
        k=60,
    )


def test_search_passes_configured_weights_and_rrf_k_to_fusion() -> None:
    service, _, _, vector_store = create_service(
        dense_weight=2.0,
        sparse_weight=0.75,
        rrf_k=40,
    )
    fused_result = create_result("d" * 64)

    with patch(
        "app.services.hybrid_search.reciprocal_rank_fusion",
        return_value=[fused_result],
    ) as fusion:
        service.search(
            HybridSearchRequest(
                query="remote work policy",
                limit=3,
                candidate_limit=10,
            )
        )

    fusion.assert_called_once_with(
        [
            vector_store.search_dense.return_value,
            vector_store.search_sparse.return_value,
        ],
        weights=[2.0, 0.75],
        limit=3,
        k=40,
    )


def test_service_defaults_to_evaluated_weights() -> None:
    service, _, _, _ = create_service()

    assert service.dense_weight == 1.5
    assert service.sparse_weight == 1.0
    assert service.rrf_k == 60


@pytest.mark.parametrize("dense_weight", [0.0, -1.0])
def test_service_rejects_non_positive_dense_weight(dense_weight: float) -> None:
    with pytest.raises(ValueError, match="hybrid weights must be greater than zero"):
        create_service(dense_weight=dense_weight)


@pytest.mark.parametrize("dense_weight", [float("nan"), float("inf")])
def test_service_rejects_non_finite_dense_weight(dense_weight: float) -> None:
    with pytest.raises(ValueError, match="hybrid weights must be finite"):
        create_service(dense_weight=dense_weight)


@pytest.mark.parametrize("sparse_weight", [0.0, -1.0])
def test_service_rejects_non_positive_sparse_weight(sparse_weight: float) -> None:
    with pytest.raises(ValueError, match="hybrid weights must be greater than zero"):
        create_service(sparse_weight=sparse_weight)


@pytest.mark.parametrize("sparse_weight", [float("nan"), float("inf")])
def test_service_rejects_non_finite_sparse_weight(sparse_weight: float) -> None:
    with pytest.raises(ValueError, match="hybrid weights must be finite"):
        create_service(sparse_weight=sparse_weight)


def test_service_rejects_invalid_rrf_k() -> None:
    with pytest.raises(ValueError, match="rrf_k must be greater than zero"):
        create_service(rrf_k=0)


@pytest.mark.parametrize(
    ("dense_results", "sparse_results"),
    [
        ([], [create_result("c" * 64)]),
        ([create_result("b" * 64)], []),
        ([], []),
    ],
)
def test_search_supports_empty_candidate_lists(
    dense_results: list[DenseSearchResult],
    sparse_results: list[DenseSearchResult],
) -> None:
    service, _, _, vector_store = create_service()
    vector_store.search_dense.return_value = dense_results
    vector_store.search_sparse.return_value = sparse_results

    response = service.search(HybridSearchRequest(query="remote work"))

    assert response.result_count == len(response.results)


def test_search_rejects_whitespace_query() -> None:
    service, embedding_provider, sparse_embedding_provider, vector_store = (
        create_service()
    )
    request = HybridSearchRequest.model_construct(
        query="   ",
        limit=5,
        candidate_limit=20,
        document_id=None,
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        service.search(request)

    embedding_provider.embed_query.assert_not_called()
    sparse_embedding_provider.embed_query.assert_not_called()
    vector_store.search_dense.assert_not_called()
    vector_store.search_sparse.assert_not_called()
