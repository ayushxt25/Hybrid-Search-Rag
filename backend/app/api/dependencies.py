from functools import lru_cache

from app.context.assembler import ContextAssembler
from app.core.config import get_settings
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.prompting.builder import GroundedPromptBuilder
from app.services.dense_search import DenseSearchService
from app.services.document_indexing import DocumentIndexingService
from app.services.hybrid_search import HybridSearchService
from app.services.sparse_search import SparseSearchService
from app.sparse.hashed_lexical import HashedLexicalSparseProvider
from app.vectorstore.qdrant import QdrantVectorStore


@lru_cache
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the shared dense-embedding provider."""
    return SentenceTransformerEmbeddingProvider()


@lru_cache
def get_sparse_embedding_provider() -> HashedLexicalSparseProvider:
    """Return the shared sparse lexical embedding provider."""
    return HashedLexicalSparseProvider()


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

    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_hybrid_collection_name,
        vector_dimensions=settings.dense_embedding_dimensions,
        sparse_enabled=True,
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

    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_hybrid_collection_name,
        vector_dimensions=settings.dense_embedding_dimensions,
        sparse_enabled=True,
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

    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_hybrid_collection_name,
        vector_dimensions=settings.dense_embedding_dimensions,
        sparse_enabled=True,
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

    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_hybrid_collection_name,
        vector_dimensions=settings.dense_embedding_dimensions,
        sparse_enabled=True,
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
