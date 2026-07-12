from functools import lru_cache

from app.core.config import get_settings
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.services.document_indexing import DocumentIndexingService
from app.vectorstore.qdrant import QdrantVectorStore


@lru_cache
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the shared dense-embedding provider."""
    return SentenceTransformerEmbeddingProvider()


@lru_cache
def get_document_indexing_service() -> DocumentIndexingService:
    """Return the shared document-indexing service."""
    settings = get_settings()
    embedding_provider = get_embedding_provider()

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
        collection_name=settings.qdrant_collection_name,
        vector_dimensions=settings.dense_embedding_dimensions,
    )

    return DocumentIndexingService(
        ingestion_pipeline=ingestion_pipeline,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
