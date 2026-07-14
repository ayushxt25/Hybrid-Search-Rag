from unittest.mock import Mock, patch

import pytest

from app.api.dependencies import (
    get_context_assembler,
    get_dense_search_service,
    get_document_indexing_service,
    get_embedding_provider,
    get_hybrid_search_service,
    get_sparse_embedding_provider,
    get_sparse_search_service,
)
from app.context.assembler import ContextAssembler
from app.services.dense_search import DenseSearchService
from app.services.document_indexing import DocumentIndexingService
from app.services.hybrid_search import HybridSearchService
from app.services.sparse_search import SparseSearchService


@pytest.fixture(autouse=True)
def clear_dependency_caches():
    get_embedding_provider.cache_clear()
    get_sparse_embedding_provider.cache_clear()
    get_document_indexing_service.cache_clear()
    get_dense_search_service.cache_clear()
    get_sparse_search_service.cache_clear()
    get_hybrid_search_service.cache_clear()
    get_context_assembler.cache_clear()

    yield

    get_embedding_provider.cache_clear()
    get_sparse_embedding_provider.cache_clear()
    get_document_indexing_service.cache_clear()
    get_dense_search_service.cache_clear()
    get_sparse_search_service.cache_clear()
    get_hybrid_search_service.cache_clear()
    get_context_assembler.cache_clear()


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


@patch("app.api.dependencies.HashedLexicalSparseProvider")
def test_sparse_embedding_provider_is_cached(
    provider_class: Mock,
) -> None:
    provider = Mock()
    provider_class.return_value = provider

    first = get_sparse_embedding_provider()
    second = get_sparse_embedding_provider()

    assert first is provider
    assert second is provider
    provider_class.assert_called_once_with()


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.DocumentIngestionPipeline")
@patch("app.api.dependencies.get_sparse_embedding_provider")
@patch("app.api.dependencies.get_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_document_indexing_service_is_created_and_cached(
    settings_factory: Mock,
    provider_factory: Mock,
    sparse_provider_factory: Mock,
    pipeline_class: Mock,
    vector_store_class: Mock,
) -> None:
    settings = Mock(
        qdrant_url="http://localhost:6333",
        qdrant_collection_name="test_chunks",
        qdrant_hybrid_collection_name="test_hybrid_chunks",
        dense_embedding_dimensions=384,
    )
    provider = Mock(dimensions=384)
    sparse_provider = Mock()
    pipeline = Mock()
    vector_store = Mock()

    settings_factory.return_value = settings
    provider_factory.return_value = provider
    sparse_provider_factory.return_value = sparse_provider
    pipeline_class.return_value = pipeline
    vector_store_class.return_value = vector_store

    first = get_document_indexing_service()
    second = get_document_indexing_service()

    assert isinstance(first, DocumentIndexingService)
    assert second is first
    assert first.ingestion_pipeline is pipeline
    assert first.embedding_provider is provider
    assert first.sparse_embedding_provider is sparse_provider
    assert first.vector_store is vector_store

    pipeline_class.assert_called_once_with(
        chunk_size=200,
        chunk_overlap=40,
    )
    vector_store_class.assert_called_once_with(
        url="http://localhost:6333",
        collection_name="test_hybrid_chunks",
        vector_dimensions=384,
        sparse_enabled=True,
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


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.get_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_dense_search_service_is_created_and_cached(
    settings_factory: Mock,
    provider_factory: Mock,
    vector_store_class: Mock,
) -> None:
    settings = Mock(
        qdrant_url="http://localhost:6333",
        qdrant_collection_name="test_chunks",
        qdrant_hybrid_collection_name="test_hybrid_chunks",
        dense_embedding_dimensions=384,
    )
    provider = Mock(dimensions=384)
    vector_store = Mock()

    settings_factory.return_value = settings
    provider_factory.return_value = provider
    vector_store_class.return_value = vector_store

    first = get_dense_search_service()
    second = get_dense_search_service()

    assert isinstance(first, DenseSearchService)
    assert second is first
    assert first.embedding_provider is provider
    assert first.vector_store is vector_store

    vector_store_class.assert_called_once_with(
        url="http://localhost:6333",
        collection_name="test_hybrid_chunks",
        vector_dimensions=384,
        sparse_enabled=True,
    )


@patch("app.api.dependencies.get_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_dense_search_dependency_rejects_dimension_mismatch(
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
        get_dense_search_service()


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.get_sparse_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_sparse_search_service_is_created_and_cached(
    settings_factory: Mock,
    sparse_provider_factory: Mock,
    vector_store_class: Mock,
) -> None:
    settings = Mock(
        qdrant_url="http://localhost:6333",
        qdrant_hybrid_collection_name="test_hybrid_chunks",
        dense_embedding_dimensions=384,
    )
    sparse_provider = Mock()
    vector_store = Mock()

    settings_factory.return_value = settings
    sparse_provider_factory.return_value = sparse_provider
    vector_store_class.return_value = vector_store

    first = get_sparse_search_service()
    second = get_sparse_search_service()

    assert isinstance(first, SparseSearchService)
    assert second is first
    assert first.sparse_embedding_provider is sparse_provider
    assert first.vector_store is vector_store

    vector_store_class.assert_called_once_with(
        url="http://localhost:6333",
        collection_name="test_hybrid_chunks",
        vector_dimensions=384,
        sparse_enabled=True,
    )


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.get_sparse_embedding_provider")
@patch("app.api.dependencies.get_embedding_provider")
@patch("app.api.dependencies.get_settings")
def test_hybrid_search_service_is_created_and_cached(
    settings_factory: Mock,
    provider_factory: Mock,
    sparse_provider_factory: Mock,
    vector_store_class: Mock,
) -> None:
    settings = Mock(
        qdrant_url="http://localhost:6333",
        qdrant_hybrid_collection_name="test_hybrid_chunks",
        dense_embedding_dimensions=384,
        hybrid_dense_weight=2.0,
        hybrid_sparse_weight=0.75,
        hybrid_rrf_k=40,
    )
    provider = Mock()
    sparse_provider = Mock()
    vector_store = Mock()

    settings_factory.return_value = settings
    provider_factory.return_value = provider
    sparse_provider_factory.return_value = sparse_provider
    vector_store_class.return_value = vector_store

    first = get_hybrid_search_service()
    second = get_hybrid_search_service()

    assert isinstance(first, HybridSearchService)
    assert second is first
    assert first.embedding_provider is provider
    assert first.sparse_embedding_provider is sparse_provider
    assert first.vector_store is vector_store
    assert first.dense_weight == 2.0
    assert first.sparse_weight == 0.75
    assert first.rrf_k == 40

    vector_store_class.assert_called_once_with(
        url="http://localhost:6333",
        collection_name="test_hybrid_chunks",
        vector_dimensions=384,
        sparse_enabled=True,
    )


@patch("app.api.dependencies.get_settings")
def test_context_assembler_is_created_and_cached(settings_factory: Mock) -> None:
    settings_factory.return_value = Mock(
        context_max_characters=5000,
        context_max_sources=4,
        context_include_metadata_headers=False,
    )

    first = get_context_assembler()
    second = get_context_assembler()

    assert isinstance(first, ContextAssembler)
    assert second is first
    assert first.max_characters == 5000
    assert first.max_sources == 4
    assert first.include_metadata_headers is False
