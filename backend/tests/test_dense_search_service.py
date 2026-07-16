from unittest.mock import Mock

import pytest

from app.embeddings.base import DenseEmbeddingProvider
from app.retrieval.filters import RetrievalFilters
from app.schemas.embedding import QueryEmbedding
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import DenseSearchRequest
from app.services.dense_search import DenseSearchService
from app.vectorstore.qdrant import QdrantVectorStore

DOCUMENT_ID = "a" * 64
CHUNK_ID = "b" * 64


def create_service(
    *,
    embedding_dimensions: int = 384,
    vector_dimensions: int = 384,
) -> tuple[DenseSearchService, Mock, Mock]:
    embedding_provider = Mock(spec=DenseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    embedding_provider.embed_query.return_value = QueryEmbedding(
        query="remote work policy",
        vector=[0.0] * embedding_dimensions,
        dimensions=embedding_dimensions,
    )

    vector_store.vector_dimensions = vector_dimensions
    vector_store.search_dense.return_value = [
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

    service = DenseSearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return service, embedding_provider, vector_store


def test_search_embeds_query_and_returns_results() -> None:
    service, embedding_provider, vector_store = create_service()

    request = DenseSearchRequest(
        query="  remote work policy  ",
        limit=3,
        score_threshold=0.5,
        document_id=DOCUMENT_ID,
    )

    response = service.search(request)

    assert response.query == "remote work policy"
    assert response.result_count == 1
    assert response.results[0].chunk_id == CHUNK_ID

    embedding_provider.embed_query.assert_called_once_with("remote work policy")

    vector_store.search_dense.assert_called_once_with(
        query_vector=[0.0] * 384,
        limit=3,
        score_threshold=0.5,
        filters=RetrievalFilters(document_ids=[DOCUMENT_ID]),
    )


def test_search_forwards_filters() -> None:
    service, _, vector_store = create_service()

    service.search(
        DenseSearchRequest(
            query="remote work",
            document_ids=[DOCUMENT_ID, DOCUMENT_ID],
            content_types=["TEXT/PLAIN"],
        )
    )

    assert vector_store.search_dense.call_args.kwargs["filters"] == RetrievalFilters(
        document_ids=[DOCUMENT_ID],
        content_types=["text/plain"],
    )


def test_search_returns_empty_results() -> None:
    service, _, vector_store = create_service()
    vector_store.search_dense.return_value = []

    response = service.search(DenseSearchRequest(query="unknown internal policy"))

    assert response.result_count == 0
    assert response.results == []


def test_search_rejects_whitespace_query() -> None:
    service, embedding_provider, vector_store = create_service()

    request = DenseSearchRequest.model_construct(
        query="   ",
        limit=5,
        score_threshold=None,
        document_id=None,
        document_ids=None,
        content_types=None,
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        service.search(request)

    embedding_provider.embed_query.assert_not_called()
    vector_store.search_dense.assert_not_called()


def test_search_rejects_dimension_mismatch() -> None:
    service, _, vector_store = create_service(
        embedding_dimensions=768,
        vector_dimensions=384,
    )

    with pytest.raises(
        RuntimeError,
        match="dimensions do not match",
    ):
        service.search(DenseSearchRequest(query="remote work"))

    vector_store.search_dense.assert_not_called()
