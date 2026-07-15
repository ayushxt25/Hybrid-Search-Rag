from unittest.mock import Mock, patch

import pytest

from app.api.dependencies import (
    get_context_assembler,
    get_dense_search_service,
    get_document_indexing_service,
    get_embedding_provider,
    get_generation_provider,
    get_grounded_answer_rate_limiter,
    get_grounded_answer_service,
    get_grounded_prompt_builder,
    get_hybrid_search_service,
    get_readiness_qdrant_vector_store,
    get_readiness_service,
    get_sparse_embedding_provider,
    get_sparse_search_service,
    shutdown_dependencies,
)
from app.context.assembler import ContextAssembler
from app.generation.openai import OpenAIGenerationProvider
from app.generation.service import GroundedAnswerService
from app.prompting.builder import GroundedPromptBuilder
from app.rate_limit.in_memory import InMemoryFixedWindowRateLimiter
from app.services.dense_search import DenseSearchService
from app.services.document_indexing import DocumentIndexingService
from app.services.hybrid_search import HybridSearchService
from app.services.sparse_search import SparseSearchService
from app.vectorstore.qdrant import QdrantVectorStore


@pytest.fixture(autouse=True)
def clear_dependency_caches():
    shutdown_dependencies()

    yield

    shutdown_dependencies()


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


@patch("app.api.dependencies.get_settings")
def test_grounded_prompt_builder_is_created_and_cached(
    settings_factory: Mock,
) -> None:
    settings_factory.return_value = Mock(
        prompt_max_question_characters=1500,
        prompt_require_citations=False,
        prompt_allow_general_knowledge=True,
    )

    first = get_grounded_prompt_builder()
    second = get_grounded_prompt_builder()

    assert isinstance(first, GroundedPromptBuilder)
    assert second is first
    assert first.max_question_characters == 1500
    assert first.require_citations is False
    assert first.allow_general_knowledge is True


@patch("app.api.dependencies.OpenAIGenerationProvider")
@patch("app.api.dependencies.get_settings")
def test_generation_provider_is_created_and_cached(
    settings_factory: Mock,
    provider_class: Mock,
) -> None:
    settings_factory.return_value = Mock(
        openai_api_key="test-key",
        openai_generation_model="gpt-test",
        openai_base_url="https://example.test/v1",
        openai_generation_timeout_seconds=12.5,
        openai_generation_max_retries=4,
    )
    provider = Mock(spec=OpenAIGenerationProvider)
    provider_class.return_value = provider

    first = get_generation_provider()
    second = get_generation_provider()

    assert first is provider
    assert second is provider
    provider_class.assert_called_once_with(
        api_key="test-key",
        model_name="gpt-test",
        base_url="https://example.test/v1",
        timeout_seconds=12.5,
        max_retries=4,
    )


@patch("app.api.dependencies.OpenAIGenerationProvider")
def test_openai_provider_is_not_created_at_module_import(
    provider_class: Mock,
) -> None:
    provider_class.assert_not_called()


@patch("app.api.dependencies.get_generation_provider")
@patch("app.api.dependencies.get_grounded_prompt_builder")
@patch("app.api.dependencies.get_context_assembler")
@patch("app.api.dependencies.get_hybrid_search_service")
@patch("app.api.dependencies.get_settings")
def test_grounded_answer_service_is_created_and_cached(
    settings_factory: Mock,
    search_factory: Mock,
    assembler_factory: Mock,
    prompt_builder_factory: Mock,
    provider_factory: Mock,
) -> None:
    settings_factory.return_value = Mock(generation_require_answer_citations=False)
    search_service = Mock()
    assembler = Mock()
    prompt_builder = Mock()
    provider = Mock()
    search_factory.return_value = search_service
    assembler_factory.return_value = assembler
    prompt_builder_factory.return_value = prompt_builder
    provider_factory.return_value = provider

    first = get_grounded_answer_service()
    second = get_grounded_answer_service()

    assert isinstance(first, GroundedAnswerService)
    assert second is first
    assert first.hybrid_search_service is search_service
    assert first.context_assembler is assembler
    assert first.prompt_builder is prompt_builder
    assert first.generation_provider is provider
    assert first.require_answer_citations is False


@patch("app.api.dependencies.get_settings")
def test_grounded_answer_rate_limiter_is_created_and_cached(
    settings_factory: Mock,
) -> None:
    settings_factory.return_value = Mock(
        grounded_answer_rate_limit_requests=3,
        grounded_answer_rate_limit_window_seconds=9,
    )

    first = get_grounded_answer_rate_limiter()
    second = get_grounded_answer_rate_limiter()

    assert isinstance(first, InMemoryFixedWindowRateLimiter)
    assert second is first
    assert first.limit == 3
    assert first.window_seconds == 9


@patch("app.api.dependencies.QdrantVectorStore")
@patch("app.api.dependencies.get_settings")
def test_readiness_qdrant_vector_store_is_created_and_cached(
    settings_factory: Mock,
    vector_store_class: Mock,
) -> None:
    settings_factory.return_value = Mock(
        qdrant_url="http://localhost:6333",
        qdrant_hybrid_collection_name="test_hybrid_chunks",
        dense_embedding_dimensions=384,
        qdrant_health_timeout_seconds=1.5,
    )
    vector_store = Mock(spec=QdrantVectorStore)
    vector_store_class.return_value = vector_store

    first = get_readiness_qdrant_vector_store()
    second = get_readiness_qdrant_vector_store()

    assert first is vector_store
    assert second is vector_store
    vector_store_class.assert_called_once_with(
        url="http://localhost:6333",
        collection_name="test_hybrid_chunks",
        vector_dimensions=384,
        sparse_enabled=True,
        timeout_seconds=1.5,
    )


@patch("app.api.dependencies.get_readiness_qdrant_vector_store")
@patch("app.api.dependencies.get_settings")
def test_readiness_service_is_created_and_cached(
    settings_factory: Mock,
    qdrant_factory: Mock,
) -> None:
    settings = Mock()
    qdrant_checker = Mock()
    settings_factory.return_value = settings
    qdrant_factory.return_value = qdrant_checker

    first = get_readiness_service()
    second = get_readiness_service()

    assert second is first
    assert first.settings is settings
    assert first.qdrant_checker is qdrant_checker


@patch("app.api.dependencies.QdrantVectorStore")
def test_readiness_dependency_is_lazy(vector_store_class: Mock) -> None:
    vector_store_class.assert_not_called()
