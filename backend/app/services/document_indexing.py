from pathlib import Path

from app.embeddings.base import DenseEmbeddingProvider
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.schemas.document import IngestedDocument
from app.schemas.indexing import IndexedDocumentResult
from app.sparse.base import SparseEmbeddingProvider
from app.vectorstore.qdrant import QdrantVectorStore


class DocumentIndexingService:
    """Coordinate document ingestion, embedding, and vector storage."""

    def __init__(
        self,
        *,
        ingestion_pipeline: DocumentIngestionPipeline,
        embedding_provider: DenseEmbeddingProvider,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.ingestion_pipeline = ingestion_pipeline
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_store = vector_store

    def index_document(
        self,
        document_path: Path,
    ) -> IndexedDocumentResult:
        """Ingest and persist one document in the vector database."""
        ingested_document = self.index_document_for_internal_use(document_path)
        document = ingested_document.document

        return IndexedDocumentResult(
            document_id=document.document_id,
            content_hash=document.content_hash,
            file_name=document.file_name,
            file_extension=document.file_extension,
            chunk_count=ingested_document.chunk_count,
            indexed_points=ingested_document.chunk_count,
        )

    def replace_document(
        self,
        *,
        document_path: Path,
        replace_document_id: str,
    ) -> IndexedDocumentResult:
        """Prepare a new document, then replace old chunks immediately before upsert."""
        result, _ = self.replace_document_for_internal_use(
            document_path=document_path,
            replace_document_id=replace_document_id,
        )
        return result

    def replace_document_for_internal_use(
        self,
        *,
        document_path: Path,
        replace_document_id: str,
    ) -> tuple[IndexedDocumentResult, int]:
        """Replace one document and return the public result plus deleted count."""
        ingested_document = self.ingestion_pipeline.ingest(document_path)
        document = ingested_document.document

        dense_embeddings = self.embedding_provider.embed_chunks(
            ingested_document.chunks
        )
        sparse_embeddings = self.sparse_embedding_provider.embed_chunks(
            ingested_document.chunks
        )

        self._validate_embedding_counts(
            ingested_document=ingested_document,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        self.vector_store.ensure_collection()

        deleted_chunks, indexed_points = self.vector_store.replace_document(
            replace_document_id=replace_document_id,
            ingested_document=ingested_document,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        if indexed_points != ingested_document.chunk_count:
            raise RuntimeError(
                "Indexed point count does not match document chunk count."
            )

        return (
            IndexedDocumentResult(
                document_id=document.document_id,
                content_hash=document.content_hash,
                file_name=document.file_name,
                file_extension=document.file_extension,
                chunk_count=ingested_document.chunk_count,
                indexed_points=indexed_points,
            ),
            deleted_chunks,
        )

    def index_document_for_internal_use(
        self,
        document_path: Path,
    ) -> IngestedDocument:
        """Ingest, index, and return chunk metadata for internal tooling."""
        ingested_document = self.ingestion_pipeline.ingest(document_path)

        dense_embeddings = self.embedding_provider.embed_chunks(
            ingested_document.chunks
        )
        sparse_embeddings = self.sparse_embedding_provider.embed_chunks(
            ingested_document.chunks
        )

        self._validate_embedding_counts(
            ingested_document=ingested_document,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        self.vector_store.ensure_collection()

        indexed_points = self.vector_store.upsert_hybrid_document(
            ingested_document=ingested_document,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        if indexed_points != ingested_document.chunk_count:
            raise RuntimeError(
                "Indexed point count does not match document chunk count."
            )

        return ingested_document

    @staticmethod
    def _validate_embedding_counts(
        *,
        ingested_document: IngestedDocument,
        dense_embeddings,
        sparse_embeddings,
    ) -> None:
        if len(dense_embeddings) != ingested_document.chunk_count:
            raise RuntimeError(
                "Dense embedding count does not match document chunk count."
            )

        if len(sparse_embeddings) != ingested_document.chunk_count:
            raise RuntimeError(
                "Sparse embedding count does not match document chunk count."
            )
