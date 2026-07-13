from pathlib import Path
from unittest.mock import Mock

import pytest

from app.embeddings.base import DenseEmbeddingProvider
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.services.document_indexing import DocumentIndexingService
from app.sparse.base import SparseEmbeddingProvider
from app.vectorstore.qdrant import QdrantVectorStore


@pytest.fixture
def sample_document_path(
    tmp_path: Path,
) -> Path:
    document_path = tmp_path / "remote_policy.txt"

    document_path.write_text(
        (
            "Employees may work remotely for three days per week. "
            "Manager approval is required for remote work."
        ),
        encoding="utf-8",
    )

    return document_path


def create_service(
    *,
    embedding_provider: Mock,
    sparse_embedding_provider: Mock,
    vector_store: Mock,
) -> DocumentIndexingService:
    pipeline = DocumentIngestionPipeline(
        chunk_size=8,
        chunk_overlap=2,
    )

    return DocumentIndexingService(
        ingestion_pipeline=pipeline,
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )


def test_index_document_coordinates_complete_pipeline(
    sample_document_path: Path,
) -> None:
    embedding_provider = Mock(spec=DenseEmbeddingProvider)
    sparse_embedding_provider = Mock(spec=SparseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    embedding_provider.embed_chunks.side_effect = lambda chunks: [
        Mock(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            dimensions=384,
            vector=[0.0] * 384,
        )
        for chunk in chunks
    ]
    sparse_embedding_provider.embed_chunks.side_effect = lambda chunks: [
        Mock(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            indices=[chunk.chunk_index],
            values=[1.0],
        )
        for chunk in chunks
    ]

    def count_embeddings(
        *,
        ingested_document,
        dense_embeddings,
        sparse_embeddings,
    ) -> int:
        assert len(sparse_embeddings) == len(dense_embeddings)
        return len(dense_embeddings)

    vector_store.upsert_hybrid_document.side_effect = count_embeddings

    service = create_service(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )

    result = service.index_document(sample_document_path)

    assert result.file_name == "remote_policy.txt"
    assert result.file_extension == ".txt"
    assert result.chunk_count > 0
    assert result.indexed_points == result.chunk_count
    assert len(result.document_id) == 64
    assert len(result.content_hash) == 64

    embedding_provider.embed_chunks.assert_called_once()
    sparse_embedding_provider.embed_chunks.assert_called_once()
    vector_store.ensure_collection.assert_called_once()
    vector_store.upsert_hybrid_document.assert_called_once()


def test_index_document_rejects_dense_embedding_count_mismatch(
    sample_document_path: Path,
) -> None:
    embedding_provider = Mock(spec=DenseEmbeddingProvider)
    sparse_embedding_provider = Mock(spec=SparseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    embedding_provider.embed_chunks.return_value = []
    sparse_embedding_provider.embed_chunks.side_effect = lambda chunks: [
        Mock(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            indices=[chunk.chunk_index],
            values=[1.0],
        )
        for chunk in chunks
    ]

    service = create_service(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Dense embedding count does not match",
    ):
        service.index_document(sample_document_path)

    vector_store.ensure_collection.assert_not_called()
    vector_store.upsert_hybrid_document.assert_not_called()


def test_index_document_rejects_sparse_embedding_count_mismatch(
    sample_document_path: Path,
) -> None:
    embedding_provider = Mock(spec=DenseEmbeddingProvider)
    sparse_embedding_provider = Mock(spec=SparseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    embedding_provider.embed_chunks.side_effect = lambda chunks: [
        Mock(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            dimensions=384,
            vector=[0.0] * 384,
        )
        for chunk in chunks
    ]
    sparse_embedding_provider.embed_chunks.return_value = []

    service = create_service(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Sparse embedding count does not match",
    ):
        service.index_document(sample_document_path)

    vector_store.ensure_collection.assert_not_called()
    vector_store.upsert_hybrid_document.assert_not_called()


def test_index_document_rejects_indexed_point_mismatch(
    sample_document_path: Path,
) -> None:
    embedding_provider = Mock(spec=DenseEmbeddingProvider)
    sparse_embedding_provider = Mock(spec=SparseEmbeddingProvider)
    vector_store = Mock(spec=QdrantVectorStore)

    embedding_provider.embed_chunks.side_effect = lambda chunks: [
        Mock(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            dimensions=384,
            vector=[0.0] * 384,
        )
        for chunk in chunks
    ]
    sparse_embedding_provider.embed_chunks.side_effect = lambda chunks: [
        Mock(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            indices=[chunk.chunk_index],
            values=[1.0],
        )
        for chunk in chunks
    ]

    vector_store.upsert_hybrid_document.return_value = 0

    service = create_service(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Indexed point count does not match",
    ):
        service.index_document(sample_document_path)

    vector_store.ensure_collection.assert_called_once()
    vector_store.upsert_hybrid_document.assert_called_once()
