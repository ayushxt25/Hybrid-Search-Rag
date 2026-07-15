from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_readiness_service
from app.health.service import ReadinessService
from app.main import app
from app.schemas.health import ComponentHealth, ReadinessResponse
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
)

client = TestClient(app)


class StubReadinessService:
    def __init__(self, response: ReadinessResponse) -> None:
        self.response = response

    def check(self) -> ReadinessResponse:
        return self.response


class FakeQdrantChecker:
    def __init__(
        self,
        *,
        vector_dimensions: int = 384,
        error: Exception | None = None,
    ) -> None:
        self.vector_dimensions = vector_dimensions
        self.error = error
        self.checks = 0

    def check_readiness(self) -> None:
        self.checks += 1
        if self.error is not None:
            raise self.error


def settings(**overrides):
    values = {
        "readiness_enabled": True,
        "observability_enabled": True,
        "openai_generation_model": "gpt-test",
        "openai_api_key": "test-key",
        "dense_embedding_dimensions": 384,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def clear_overrides():
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def override_readiness(response: ReadinessResponse) -> None:
    app.dependency_overrides[get_readiness_service] = lambda: StubReadinessService(
        response
    )


def test_health_check_returns_service_information() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "Hybrid Search RAG",
        "version": "0.1.0",
        "environment": "development",
    }


def test_liveness_returns_alive() -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_liveness_creates_no_external_clients_or_heavy_dependencies() -> None:
    with (
        patch("app.api.dependencies.QdrantVectorStore") as vector_store_class,
        patch("app.api.dependencies.OpenAIGenerationProvider") as provider_class,
        patch(
            "app.api.dependencies.SentenceTransformerEmbeddingProvider"
        ) as embedding_class,
    ):
        response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    vector_store_class.assert_not_called()
    provider_class.assert_not_called()
    embedding_class.assert_not_called()


def test_readiness_returns_200_when_all_components_are_healthy() -> None:
    override_readiness(
        ReadinessResponse(
            status="ready",
            components={
                "qdrant": ComponentHealth(status="healthy"),
                "generation": ComponentHealth(status="healthy"),
                "embedding_configuration": ComponentHealth(status="healthy"),
            },
        )
    )

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (
            VectorStoreConnectionError("raw secret qdrant url"),
            "Vector database is unavailable.",
        ),
        (
            VectorStoreConfigurationError("raw missing collection payload"),
            "Required hybrid collection is unavailable or incompatible.",
        ),
    ],
)
def test_readiness_returns_503_for_qdrant_failures(
    error: Exception,
    detail: str,
) -> None:
    service = ReadinessService(
        settings=settings(),
        qdrant_checker=FakeQdrantChecker(error=error),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["components"]["qdrant"] == {
        "status": "unhealthy",
        "detail": detail,
    }
    assert "raw secret" not in response.text
    assert "raw missing" not in response.text


def test_readiness_returns_503_when_collection_is_incompatible() -> None:
    service = ReadinessService(
        settings=settings(),
        qdrant_checker=FakeQdrantChecker(
            error=VectorStoreConfigurationError("dense vector leaked")
        ),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["components"]["qdrant"]["detail"] == (
        "Required hybrid collection is unavailable or incompatible."
    )
    assert "dense vector leaked" not in response.text


def test_readiness_returns_503_when_openai_key_is_missing() -> None:
    service = ReadinessService(
        settings=settings(openai_api_key=" "),
        qdrant_checker=FakeQdrantChecker(),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["components"]["generation"] == {
        "status": "unhealthy",
        "detail": "Generation provider is not configured.",
    }


def test_readiness_does_not_call_openai_or_load_embedding_model() -> None:
    service = ReadinessService(
        settings=settings(),
        qdrant_checker=FakeQdrantChecker(),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    with (
        patch("app.api.dependencies.OpenAIGenerationProvider") as provider_class,
        patch(
            "app.api.dependencies.SentenceTransformerEmbeddingProvider"
        ) as embedding_class,
    ):
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    provider_class.assert_not_called()
    embedding_class.assert_not_called()


def test_readiness_detects_embedding_dimension_mismatch() -> None:
    service = ReadinessService(
        settings=settings(dense_embedding_dimensions=768),
        qdrant_checker=FakeQdrantChecker(vector_dimensions=384),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["components"]["embedding_configuration"] == {
        "status": "unhealthy",
        "detail": "Embedding dimensions are incompatible.",
    }


def test_disabled_readiness_returns_200() -> None:
    service = ReadinessService(
        settings=settings(readiness_enabled=False),
        qdrant_checker=FakeQdrantChecker(error=RuntimeError("not called")),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "components": {
            "readiness": {
                "status": "not_configured",
                "detail": "Readiness checks are disabled.",
            }
        },
    }


def test_readiness_logging_contains_allowed_fields_only(caplog) -> None:
    service = ReadinessService(
        settings=settings(),
        qdrant_checker=FakeQdrantChecker(),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    with caplog.at_level("INFO", logger="app.health"):
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "readiness_check_completed"
    ]
    assert records
    record = records[0]
    assert record.ready is True
    assert record.qdrant_status == "healthy"
    assert record.generation_status == "healthy"
    assert record.embedding_configuration_status == "healthy"
    assert isinstance(record.elapsed_ms, float)
    assert "test-key" not in caplog.text


def test_repeated_readiness_checks_reuse_service() -> None:
    service = ReadinessService(
        settings=settings(),
        qdrant_checker=FakeQdrantChecker(),
    )
    app.dependency_overrides[get_readiness_service] = lambda: service

    assert client.get("/api/v1/health/ready").status_code == 200
    assert client.get("/api/v1/health/ready").status_code == 200
    assert service.qdrant_checker.checks == 2
