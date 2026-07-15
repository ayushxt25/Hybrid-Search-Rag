import logging
from functools import lru_cache
from typing import Protocol

from fastapi import HTTPException, Request, status

from app.context.assembler import ContextAssembler
from app.core.config import get_settings
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.generation.openai import OpenAIGenerationProvider
from app.generation.service import GroundedAnswerService
from app.health.service import ReadinessService
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.observability.request_context import get_request_id
from app.prompting.builder import GroundedPromptBuilder
from app.rate_limit.in_memory import InMemoryFixedWindowRateLimiter
from app.security.api_key import ApiKeyAuthenticator
from app.services.dense_search import DenseSearchService
from app.services.document_indexing import DocumentIndexingService
from app.services.hybrid_search import HybridSearchService
from app.services.sparse_search import SparseSearchService
from app.sparse.hashed_lexical import HashedLexicalSparseProvider
from app.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger("app.dependencies")
auth_logger = logging.getLogger("app.security")


class Closeable(Protocol):
    def close(self) -> None: ...


_registered_closeables: list[Closeable] = []
_registered_closeable_ids: set[int] = set()


def _register_closeable(resource: Closeable) -> Closeable:
    resource_id = id(resource)
    if resource_id not in _registered_closeable_ids:
        _registered_closeables.append(resource)
        _registered_closeable_ids.add(resource_id)

    return resource


@lru_cache
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the shared dense-embedding provider."""
    return SentenceTransformerEmbeddingProvider()


@lru_cache
def get_sparse_embedding_provider() -> HashedLexicalSparseProvider:
    """Return the shared sparse lexical embedding provider."""
    return HashedLexicalSparseProvider()


@lru_cache
def get_generation_provider() -> OpenAIGenerationProvider:
    """Return the shared production generation provider."""
    settings = get_settings()

    return _register_closeable(
        OpenAIGenerationProvider(
            api_key=settings.openai_api_key,
            model_name=settings.openai_generation_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_generation_timeout_seconds,
            max_retries=settings.openai_generation_max_retries,
        )
    )


@lru_cache
def get_document_indexing_service() -> DocumentIndexingService:
    """Return the shared document-indexing service."""
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    sparse_embedding_provider = get_sparse_embedding_provider()

    if embedding_provider.dimensions != settings.dense_embedding_dimensions:
        raise RuntimeError(
            "Configured dense embedding dimensions do not match the embedding model."
        )

    ingestion_pipeline = DocumentIngestionPipeline(
        chunk_size=200,
        chunk_overlap=40,
    )

    vector_store = _register_closeable(
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_hybrid_collection_name,
            vector_dimensions=settings.dense_embedding_dimensions,
            sparse_enabled=True,
        )
    )

    return DocumentIndexingService(
        ingestion_pipeline=ingestion_pipeline,
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )


@lru_cache
def get_dense_search_service() -> DenseSearchService:
    """Return the shared dense-search service."""
    settings = get_settings()
    embedding_provider = get_embedding_provider()

    if embedding_provider.dimensions != settings.dense_embedding_dimensions:
        raise RuntimeError(
            "Configured dense embedding dimensions do not match the embedding model."
        )

    vector_store = _register_closeable(
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_hybrid_collection_name,
            vector_dimensions=settings.dense_embedding_dimensions,
            sparse_enabled=True,
        )
    )

    return DenseSearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


@lru_cache
def get_sparse_search_service() -> SparseSearchService:
    """Return the shared sparse-search service."""
    settings = get_settings()
    sparse_embedding_provider = get_sparse_embedding_provider()

    vector_store = _register_closeable(
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_hybrid_collection_name,
            vector_dimensions=settings.dense_embedding_dimensions,
            sparse_enabled=True,
        )
    )

    return SparseSearchService(
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )


@lru_cache
def get_hybrid_search_service() -> HybridSearchService:
    """Return the shared hybrid-search service."""
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    sparse_embedding_provider = get_sparse_embedding_provider()

    vector_store = _register_closeable(
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_hybrid_collection_name,
            vector_dimensions=settings.dense_embedding_dimensions,
            sparse_enabled=True,
        )
    )

    return HybridSearchService(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
        dense_weight=settings.hybrid_dense_weight,
        sparse_weight=settings.hybrid_sparse_weight,
        rrf_k=settings.hybrid_rrf_k,
    )


@lru_cache
def get_context_assembler() -> ContextAssembler:
    """Return the shared retrieval-context assembler."""
    settings = get_settings()

    return ContextAssembler(
        max_characters=settings.context_max_characters,
        max_sources=settings.context_max_sources,
        include_metadata_headers=settings.context_include_metadata_headers,
    )


@lru_cache
def get_grounded_prompt_builder() -> GroundedPromptBuilder:
    """Return the shared grounded prompt builder."""
    settings = get_settings()

    return GroundedPromptBuilder(
        max_question_characters=settings.prompt_max_question_characters,
        require_citations=settings.prompt_require_citations,
        allow_general_knowledge=settings.prompt_allow_general_knowledge,
    )


@lru_cache
def get_grounded_answer_service() -> GroundedAnswerService:
    """Return the shared grounded-answer service."""
    settings = get_settings()

    return GroundedAnswerService(
        hybrid_search_service=get_hybrid_search_service(),
        context_assembler=get_context_assembler(),
        prompt_builder=get_grounded_prompt_builder(),
        generation_provider=get_generation_provider(),
        require_answer_citations=settings.generation_require_answer_citations,
    )


@lru_cache
def get_grounded_answer_rate_limiter() -> InMemoryFixedWindowRateLimiter:
    """Return the shared grounded-answer rate limiter."""
    settings = get_settings()

    return InMemoryFixedWindowRateLimiter(
        limit=settings.grounded_answer_rate_limit_requests,
        window_seconds=settings.grounded_answer_rate_limit_window_seconds,
    )


@lru_cache
def get_api_key_authenticator() -> ApiKeyAuthenticator | None:
    settings = get_settings()
    if not settings.api_auth_enabled:
        return None

    return ApiKeyAuthenticator(settings.api_auth_key_sha256 or "")


def _reject_authentication(request: Request) -> None:
    settings = get_settings()
    if settings.observability_enabled:
        route = request.scope.get("route")
        route_label = getattr(route, "path", "unknown")
        auth_logger.warning(
            "authentication_rejected",
            extra={
                "event": "authentication_rejected",
                "request_id": get_request_id(),
                "route": route_label,
                "method": request.method,
                "status_code": status.HTTP_401_UNAUTHORIZED,
            },
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Valid API credentials are required.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def require_api_key(request: Request) -> None:
    settings = get_settings()
    if not settings.api_auth_enabled:
        return

    authenticator = get_api_key_authenticator()
    if authenticator is None:
        _reject_authentication(request)

    provided_key = request.headers.get(settings.api_auth_header_name)
    if not authenticator.authenticate(provided_key):
        _reject_authentication(request)


def require_search_api_key(request: Request) -> None:
    settings = get_settings()
    if settings.api_auth_enabled and settings.api_auth_protect_search:
        require_api_key(request)


@lru_cache
def get_readiness_qdrant_vector_store() -> QdrantVectorStore:
    """Return the shared lightweight Qdrant readiness checker."""
    settings = get_settings()

    return _register_closeable(
        QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_hybrid_collection_name,
            vector_dimensions=settings.dense_embedding_dimensions,
            sparse_enabled=True,
            timeout_seconds=settings.qdrant_health_timeout_seconds,
        )
    )


@lru_cache
def get_readiness_service() -> ReadinessService:
    """Return the shared readiness service."""
    return ReadinessService(
        settings=get_settings(),
        qdrant_checker=get_readiness_qdrant_vector_store(),
    )


def shutdown_dependencies() -> None:
    errors: list[Exception] = []

    for resource in reversed(_registered_closeables):
        try:
            resource.close()
        except Exception as error:
            errors.append(error)
            logger.error(
                "dependency_shutdown_failed",
                extra={
                    "event": "dependency_shutdown_failed",
                    "resource_type": type(resource).__name__,
                    "exception_type": type(error).__name__,
                },
            )

    _registered_closeables.clear()
    _registered_closeable_ids.clear()
    get_embedding_provider.cache_clear()
    get_sparse_embedding_provider.cache_clear()
    get_document_indexing_service.cache_clear()
    get_dense_search_service.cache_clear()
    get_sparse_search_service.cache_clear()
    get_hybrid_search_service.cache_clear()
    get_context_assembler.cache_clear()
    get_grounded_prompt_builder.cache_clear()
    get_generation_provider.cache_clear()
    get_grounded_answer_service.cache_clear()
    get_grounded_answer_rate_limiter.cache_clear()
    get_api_key_authenticator.cache_clear()
    get_readiness_qdrant_vector_store.cache_clear()
    get_readiness_service.cache_clear()
    get_settings.cache_clear()
    if errors:
        return
