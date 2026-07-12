from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreDataError,
)
from app.vectorstore.qdrant import QdrantVectorStore


@pytest.fixture(scope="module")
def embedding_provider() -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider()


@pytest.fixture
def vector_store() -> QdrantVectorStore:
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_document_chunks",
        vector_dimensions=384,
    )
    store.ensure_collection()

    return store


def create_ingested_document(
    tmp_path: Path,
    *,
    file_name: str = "policy.txt",
    content: str = (
        "Employees may work remotely for three days per week. "
        "Employees receive eighteen paid leave days."
    ),
):
    document_path = tmp_path / file_name
    document_path.write_text(content, encoding="utf-8")

    pipeline = DocumentIngestionPipeline(
        chunk_size=8,
        chunk_overlap=2,
    )

    return pipeline.ingest(document_path)


def test_ensure_collection_is_idempotent(
    vector_store: QdrantVectorStore,
) -> None:
    vector_store.ensure_collection()
    vector_store.ensure_collection()


def test_upsert_and_search_dense_vectors(
    tmp_path: Path,
    vector_store: QdrantVectorStore,
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    document = create_ingested_document(tmp_path)
    embeddings = embedding_provider.embed_chunks(document.chunks)

    written_count = vector_store.upsert_document(
        ingested_document=document,
        embeddings=embeddings,
    )

    query_embedding = embedding_provider.embed_query("Can staff work from home?")

    results = vector_store.search_dense(
        query_vector=query_embedding.vector,
        limit=3,
    )

    assert written_count == document.chunk_count
    assert results
    assert results[0].document_id == document.document.document_id
    assert results[0].file_name == "policy.txt"
    assert len(results[0].chunk_id) == 64


def test_upsert_is_idempotent(
    tmp_path: Path,
    vector_store: QdrantVectorStore,
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    document = create_ingested_document(tmp_path)
    embeddings = embedding_provider.embed_chunks(document.chunks)

    vector_store.upsert_document(
        ingested_document=document,
        embeddings=embeddings,
    )
    vector_store.upsert_document(
        ingested_document=document,
        embeddings=embeddings,
    )

    query_embedding = embedding_provider.embed_query("employee policy")

    results = vector_store.search_dense(
        query_vector=query_embedding.vector,
        limit=20,
        document_id=document.document.document_id,
    )

    assert len(results) == document.chunk_count


def test_search_can_filter_by_document_id(
    tmp_path: Path,
    vector_store: QdrantVectorStore,
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    remote_document = create_ingested_document(
        tmp_path,
        file_name="remote.txt",
        content="Employees may work remotely three days per week.",
    )
    travel_document = create_ingested_document(
        tmp_path,
        file_name="travel.txt",
        content="Hotel expenses are reimbursed up to the approved limit.",
    )

    for document in (remote_document, travel_document):
        vector_store.upsert_document(
            ingested_document=document,
            embeddings=embedding_provider.embed_chunks(document.chunks),
        )

    query_embedding = embedding_provider.embed_query("What hotel costs are paid?")

    results = vector_store.search_dense(
        query_vector=query_embedding.vector,
        limit=5,
        document_id=travel_document.document.document_id,
    )

    assert results
    assert all(
        result.document_id == travel_document.document.document_id for result in results
    )


def test_delete_document_removes_its_points(
    tmp_path: Path,
    vector_store: QdrantVectorStore,
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    document = create_ingested_document(tmp_path)
    embeddings = embedding_provider.embed_chunks(document.chunks)

    vector_store.upsert_document(
        ingested_document=document,
        embeddings=embeddings,
    )

    vector_store.delete_document(document.document.document_id)

    query_embedding = embedding_provider.embed_query("remote employee policy")

    results = vector_store.search_dense(
        query_vector=query_embedding.vector,
        limit=5,
        document_id=document.document.document_id,
    )

    assert results == []


def test_upsert_rejects_embedding_count_mismatch(
    tmp_path: Path,
    vector_store: QdrantVectorStore,
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    document = create_ingested_document(tmp_path)
    embeddings = embedding_provider.embed_chunks(document.chunks)

    with pytest.raises(
        VectorStoreDataError,
        match="counts must match",
    ):
        vector_store.upsert_document(
            ingested_document=document,
            embeddings=embeddings[:-1],
        )


def test_search_rejects_wrong_vector_dimensions(
    vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(
        VectorStoreDataError,
        match="dimensions do not match",
    ):
        vector_store.search_dense(
            query_vector=[0.1, 0.2],
        )


@pytest.mark.parametrize(
    ("collection_name", "dimensions", "message"),
    [
        ("", 384, "collection_name cannot be empty"),
        ("test", 0, "vector_dimensions must be greater than zero"),
        ("test", -1, "vector_dimensions must be greater than zero"),
    ],
)
def test_vector_store_rejects_invalid_configuration(
    collection_name: str,
    dimensions: int,
    message: str,
) -> None:
    with pytest.raises(
        VectorStoreConfigurationError,
        match=message,
    ):
        QdrantVectorStore(
            client=QdrantClient(":memory:"),
            collection_name=collection_name,
            vector_dimensions=dimensions,
        )


def test_vector_store_requires_client_or_url() -> None:
    with pytest.raises(
        VectorStoreConfigurationError,
        match="Either a Qdrant client or URL must be provided",
    ):
        QdrantVectorStore(
            collection_name="test",
            vector_dimensions=384,
        )


def test_search_rejects_invalid_limit(
    vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        vector_store.search_dense(
            query_vector=[0.0] * 384,
            limit=0,
        )


def test_search_rejects_blank_document_id(
    vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        vector_store.search_dense(
            query_vector=[0.0] * 384,
            document_id=" ",
        )
