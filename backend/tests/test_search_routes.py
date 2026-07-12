from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_dense_search_service
from app.main import app
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import DenseSearchResponse
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

client = TestClient(app)

DOCUMENT_ID = "a" * 64
CHUNK_ID = "b" * 64


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def override_search_service(service: Mock) -> None:
    app.dependency_overrides[get_dense_search_service] = lambda: service


def create_search_response() -> DenseSearchResponse:
    return DenseSearchResponse(
        query="remote work policy",
        result_count=1,
        results=[
            DenseSearchResult(
                point_id=("123e4567-e89b-12d3-a456-426614174000"),
                chunk_id=CHUNK_ID,
                document_id=DOCUMENT_ID,
                score=0.91,
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
        ],
    )


def test_dense_search_returns_ranked_results() -> None:
    service = Mock()
    service.search.return_value = create_search_response()
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "remote work policy",
            "limit": 3,
            "score_threshold": 0.5,
            "document_id": DOCUMENT_ID,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "remote work policy"
    assert body["result_count"] == 1
    assert body["results"][0]["chunk_id"] == CHUNK_ID
    assert body["results"][0]["score"] == 0.91

    service.search.assert_called_once()


def test_dense_search_rejects_empty_query() -> None:
    service = Mock()
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422
    service.search.assert_not_called()


def test_dense_search_rejects_invalid_limit() -> None:
    service = Mock()
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "remote work",
            "limit": 0,
        },
    )

    assert response.status_code == 422
    service.search.assert_not_called()


def test_dense_search_maps_value_error() -> None:
    service = Mock()
    service.search.side_effect = ValueError("query cannot be empty.")
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "remote work",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "query cannot be empty."}


def test_dense_search_maps_connection_error() -> None:
    service = Mock()
    service.search.side_effect = VectorStoreConnectionError("Qdrant unavailable.")
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "remote work",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": ("The vector database is currently unavailable.")
    }


@pytest.mark.parametrize(
    "raised_error",
    [
        VectorStoreConfigurationError("Invalid collection."),
        VectorStoreDataError("Invalid stored payload."),
    ],
)
def test_dense_search_maps_vector_store_errors(
    raised_error: Exception,
) -> None:
    service = Mock()
    service.search.side_effect = raised_error
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "remote work",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": ("Dense search failed due to a vector-store error.")
    }


def test_dense_search_maps_runtime_error() -> None:
    service = Mock()
    service.search.side_effect = RuntimeError("Embedding dimensions mismatch.")
    override_search_service(service)

    response = client.post(
        "/api/v1/search/dense",
        json={
            "query": "remote work",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": ("Dense search did not complete successfully.")
    }
