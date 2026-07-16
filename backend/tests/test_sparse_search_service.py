from unittest.mock import Mock

import pytest

from app.retrieval.filters import RetrievalFilters
from app.schemas.embedding import QuerySparseEmbedding
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import SparseSearchRequest
from app.services.sparse_search import SparseSearchService
from app.sparse.base import SparseEmbeddingProvider
from app.vectorstore.qdrant import QdrantVectorStore

DOCUMENT_ID = "a" * 64
CHUNK_ID = "b" * 64


def create_service() -> tuple[SparseSearchService, Mock, Mock]:
    sparse_embedding_provider = Mock(spec=SparseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    sparse_embedding_provider.embed_query.return_value = QuerySparseEmbedding(
        query="remote work policy",
        indices=[3, 9],
        values=[1.0, 2.0],
    )
    vector_store.search_sparse.return_value = [
        DenseSearchResult(
            point_id="123e4567-e89b-12d3-a456-426614174000",
            chunk_id=CHUNK_ID,
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
    ]

    service = SparseSearchService(
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )

    return service, sparse_embedding_provider, vector_store


def test_search_encodes_query_and_returns_results() -> None:
    service, sparse_embedding_provider, vector_store = create_service()

    response = service.search(
        SparseSearchRequest(
            query="  remote work policy  ",
            limit=3,
            document_id=DOCUMENT_ID,
        )
    )

    assert response.query == "remote work policy"
    assert response.result_count == 1
    assert response.results[0].chunk_id == CHUNK_ID
    assert response.results[0].score_diagnostics is None
    sparse_embedding_provider.embed_query.assert_called_once_with("remote work policy")
    vector_store.search_sparse.assert_called_once_with(
        query_indices=[3, 9],
        query_values=[1.0, 2.0],
        limit=3,
        filters=RetrievalFilters(document_ids=[DOCUMENT_ID]),
    )


def test_search_forwards_filters() -> None:
    service, _, vector_store = create_service()

    service.search(
        SparseSearchRequest(
            query="remote work",
            document_ids=[DOCUMENT_ID],
            content_types=["text/plain"],
        )
    )

    assert vector_store.search_sparse.call_args.kwargs["filters"] == RetrievalFilters(
        document_ids=[DOCUMENT_ID],
        content_types=["text/plain"],
    )


def test_search_attaches_diagnostics_when_requested() -> None:
    service, sparse_embedding_provider, vector_store = create_service()

    response = service.search(
        SparseSearchRequest(query="remote work", include_score_diagnostics=True)
    )

    diagnostic = response.results[0].score_diagnostics
    assert diagnostic is not None
    assert diagnostic.dense.raw_score is None
    assert diagnostic.sparse.raw_score == 0.85
    assert diagnostic.sparse.rank == 1
    assert diagnostic.sparse.weight == 1.0
    assert diagnostic.sparse.rrf_contribution == 0.0
    assert diagnostic.fused_score == 0.85
    assert diagnostic.fused_rank == 1
    sparse_embedding_provider.embed_query.assert_called_once()
    vector_store.search_sparse.assert_called_once()


def test_search_returns_empty_results() -> None:
    service, _, vector_store = create_service()
    vector_store.search_sparse.return_value = []

    response = service.search(SparseSearchRequest(query="unknown policy"))

    assert response.result_count == 0
    assert response.results == []


def test_search_rejects_whitespace_query() -> None:
    service, sparse_embedding_provider, vector_store = create_service()

    request = SparseSearchRequest.model_construct(
        query="   ",
        limit=5,
        document_id=None,
        document_ids=None,
        content_types=None,
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        service.search(request)

    sparse_embedding_provider.embed_query.assert_not_called()
    vector_store.search_sparse.assert_not_called()
