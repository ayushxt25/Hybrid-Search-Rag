from unittest.mock import Mock, patch

import pytest

from app.api.dependencies import (
    get_document_indexing_service,
    get_embedding_provider,
)
from app.services.document_indexing import DocumentIndexingService


@pytest.fixture(autouse=True)
def clear_dependency_caches():
    get_embedding_provider.cache_clear()
    get_document_indexing_service.cache_clear()

    yield

    get_embedding_provider.cache_clear()
    get_document_indexing_service.cache_clear()


@patch("app.api.dependencies.SentenceTransformerEmbeddingProvider")
def test_embedding_provider_is_cached(
    provider_class: Mock,
) -> None:
    provider = Mock()
    provider_class.return_value = provider

    first = get_embedding_provider()
    second = get_embedding_provider()

    assert first is provider
    assert second is provider
    provider_class.assert_called_once_with()


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.DocumentIngestionPipeline")
@patch("app.api.dependencies.get_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_document_indexing_service_is_created_and_cached(
    settings_factory: Mock,
    provider_factory: Mock,
    pipeline_class: Mock,
    vector_store_class: Mock,
) -> None:
    settings = Mock(
        qdrant_url="http://localhost:6333",
        qdrant_collection_name="test_chunks",
        dense_embedding_dimensions=384,
    )
    provider = Mock(dimensions=384)
    pipeline = Mock()
    vector_store = Mock()

    settings_factory.return_value = settings
    provider_factory.return_value = provider
    pipeline_class.return_value = pipeline
    vector_store_class.return_value = vector_store

    first = get_document_indexing_service()
    second = get_document_indexing_service()

    assert isinstance(first, DocumentIndexingService)
    assert second is first
    assert first.ingestion_pipeline is pipeline
    assert first.embedding_provider is provider
    assert first.vector_store is vector_store

    pipeline_class.assert_called_once_with(
        chunk_size=200,
        chunk_overlap=40,
    )
    vector_store_class.assert_called_once_with(
        url="http://localhost:6333",
        collection_name="test_chunks",
        vector_dimensions=384,
    )


@patch("app.api.dependencies.get_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_dependency_rejects_dimension_mismatch(
    settings_factory: Mock,
    provider_factory: Mock,
) -> None:
    settings_factory.return_value = Mock(
        dense_embedding_dimensions=384,
    )
    provider_factory.return_value = Mock(
        dimensions=768,
    )

    with pytest.raises(
        RuntimeError,
        match="embedding dimensions do not match",
    ):
        get_document_indexing_service()
