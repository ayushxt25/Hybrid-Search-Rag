import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.api import dependencies
from app.api.dependencies import (
    get_generation_provider,
    get_grounded_answer_rate_limiter,
    get_readiness_qdrant_vector_store,
    shutdown_dependencies,
)
from app.generation.openai import OpenAIGenerationProvider
from app.main import create_application
from app.vectorstore.qdrant import QdrantVectorStore


@pytest.fixture(autouse=True)
def clean_dependencies():
    shutdown_dependencies()

    yield

    shutdown_dependencies()


class FakeClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class OrderedCloseable:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    def close(self) -> None:
        self.closed.append(self.name)


def test_shutdown_does_not_instantiate_unused_cached_dependencies() -> None:
    with (
        patch("app.api.dependencies.QdrantVectorStore") as vector_store_class,
        patch("app.api.dependencies.OpenAIGenerationProvider") as provider_class,
        patch(
            "app.api.dependencies.SentenceTransformerEmbeddingProvider"
        ) as embedding_class,
    ):
        shutdown_dependencies()

    vector_store_class.assert_not_called()
    provider_class.assert_not_called()
    embedding_class.assert_not_called()


@patch("app.vectorstore.qdrant.QdrantClient")
def test_internally_owned_qdrant_client_is_closed(qdrant_client_class: Mock) -> None:
    client = FakeClient()
    qdrant_client_class.return_value = client
    store = QdrantVectorStore(
        url="http://qdrant.test",
        collection_name="chunks",
        vector_dimensions=3,
    )

    store.close()

    assert client.close_calls == 1


def test_injected_qdrant_client_is_not_closed() -> None:
    client = FakeClient()
    store = QdrantVectorStore(
        client=client,
        collection_name="chunks",
        vector_dimensions=3,
    )

    store.close()

    assert client.close_calls == 0


@patch("app.generation.openai.OpenAI")
def test_internally_owned_openai_client_is_closed(openai_class: Mock) -> None:
    client = FakeClient()
    openai_class.return_value = client
    provider = OpenAIGenerationProvider(api_key="test-key", model_name="gpt-test")

    provider._client()
    provider.close()

    assert client.close_calls == 1


def test_injected_openai_client_is_not_closed() -> None:
    client = FakeClient()
    provider = OpenAIGenerationProvider(
        api_key="",
        model_name="gpt-test",
        client=client,
    )

    provider.close()

    assert client.close_calls == 0


@patch("app.generation.openai.OpenAI")
def test_close_is_idempotent(openai_class: Mock) -> None:
    client = FakeClient()
    openai_class.return_value = client
    provider = OpenAIGenerationProvider(api_key="test-key", model_name="gpt-test")

    provider._client()
    provider.close()
    provider.close()

    assert client.close_calls == 1


def test_registered_resources_close_in_reverse_creation_order() -> None:
    closed: list[str] = []
    dependencies._register_closeable(OrderedCloseable("first", closed))
    dependencies._register_closeable(OrderedCloseable("second", closed))

    shutdown_dependencies()

    assert closed == ["second", "first"]


def test_caches_are_cleared_after_shutdown() -> None:
    limiter = get_grounded_answer_rate_limiter()

    shutdown_dependencies()

    assert get_grounded_answer_rate_limiter() is not limiter


def test_repeated_shutdown_is_safe() -> None:
    shutdown_dependencies()
    shutdown_dependencies()


def test_failing_close_does_not_prevent_later_resources_from_closing(caplog) -> None:
    class FailingCloseable:
        def close(self) -> None:
            raise RuntimeError("secret-token-123")

    closed: list[str] = []
    dependencies._register_closeable(OrderedCloseable("later", closed))
    dependencies._register_closeable(FailingCloseable())

    with caplog.at_level("ERROR", logger="app.dependencies"):
        shutdown_dependencies()

    assert closed == ["later"]
    assert any(
        getattr(record, "event", None) == "dependency_shutdown_failed"
        and getattr(record, "resource_type", None) == "FailingCloseable"
        and getattr(record, "exception_type", None) == "RuntimeError"
        for record in caplog.records
    )
    assert "secret-token-123" not in caplog.text


def test_application_lifespan_calls_shutdown_dependencies() -> None:
    app = create_application()

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            shutdown.assert_not_called()

    with patch("app.main.shutdown_dependencies") as shutdown:
        asyncio.run(run_lifespan())
        shutdown.assert_called_once_with()


def test_startup_remains_lazy_and_does_not_create_external_clients() -> None:
    with (
        patch("app.main.shutdown_dependencies"),
        patch("app.api.dependencies.QdrantVectorStore") as vector_store_class,
        patch("app.api.dependencies.OpenAIGenerationProvider") as provider_class,
        patch("app.api.dependencies.SentenceTransformerEmbeddingProvider"),
    ):
        app = create_application()

    vector_store_class.assert_not_called()
    provider_class.assert_not_called()
    assert app.title == "Hybrid Search RAG"


def test_generation_provider_cache_is_cleared_after_shutdown() -> None:
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_generation_model="gpt-test",
        openai_base_url=None,
        openai_generation_timeout_seconds=30.0,
        openai_generation_max_retries=2,
    )

    with patch("app.api.dependencies.get_settings", return_value=settings):
        provider = get_generation_provider()
        shutdown_dependencies()
        assert get_generation_provider() is not provider


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.get_settings")
def test_lifecycle_shutdown_closes_readiness_client(
    settings_factory: Mock,
    vector_store_class: Mock,
) -> None:
    settings_factory.return_value = SimpleNamespace(
        qdrant_url="http://localhost:6333",
        qdrant_hybrid_collection_name="test_hybrid_chunks",
        dense_embedding_dimensions=384,
        qdrant_health_timeout_seconds=3.0,
    )
    checker = Mock()
    vector_store_class.return_value = checker

    assert get_readiness_qdrant_vector_store() is checker
    shutdown_dependencies()

    checker.close.assert_called_once_with()
