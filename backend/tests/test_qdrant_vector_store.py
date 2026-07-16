import math
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.schemas.embedding import ChunkEmbedding, ChunkSparseEmbedding
from app.sparse.hashed_lexical import HashedLexicalSparseProvider
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreDataError,
)
from app.vectorstore.identifiers import generate_qdrant_point_id
from app.vectorstore.qdrant import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantVectorStore,
)


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


@pytest.fixture
def sparse_vector_store() -> QdrantVectorStore:
    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="test_hybrid_document_chunks",
        vector_dimensions=4,
        sparse_enabled=True,
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


def create_dense_embeddings(
    document,
    *,
    dimensions: int = 4,
) -> list[ChunkEmbedding]:
    embeddings: list[ChunkEmbedding] = []

    for chunk in document.chunks:
        vector = [0.0] * dimensions
        vector[chunk.chunk_index % dimensions] = 1.0
        embeddings.append(
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                vector=vector,
                dimensions=dimensions,
            )
        )

    return embeddings


def create_sparse_embeddings(document) -> list[ChunkSparseEmbedding]:
    return HashedLexicalSparseProvider().embed_chunks(document.chunks)


def test_ensure_collection_is_idempotent(
    vector_store: QdrantVectorStore,
) -> None:
    vector_store.ensure_collection()
    vector_store.ensure_collection()


def test_sparse_enabled_collection_contains_dense_and_sparse_configs() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name="test_sparse_config",
        vector_dimensions=4,
        sparse_enabled=True,
    )

    store.ensure_collection()

    collection = client.get_collection("test_sparse_config")

    assert DENSE_VECTOR_NAME in collection.config.params.vectors
    assert SPARSE_VECTOR_NAME in collection.config.params.sparse_vectors


def test_sparse_ensure_collection_is_idempotent(
    sparse_vector_store: QdrantVectorStore,
) -> None:
    sparse_vector_store.ensure_collection()
    sparse_vector_store.ensure_collection()


def test_sparse_validation_rejects_existing_dense_only_collection() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_dense_only",
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
            )
        },
    )
    store = QdrantVectorStore(
        client=client,
        collection_name="test_dense_only",
        vector_dimensions=4,
        sparse_enabled=True,
    )

    with pytest.raises(
        VectorStoreConfigurationError,
        match="missing the named sparse vector",
    ):
        store.ensure_collection()


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


def test_hybrid_upsert_stores_both_vectors(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)
    dense_embeddings = create_dense_embeddings(document)
    sparse_embeddings = create_sparse_embeddings(document)

    written_count = sparse_vector_store.upsert_hybrid_document(
        ingested_document=document,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    point = sparse_vector_store.client.retrieve(
        collection_name=sparse_vector_store.collection_name,
        ids=[generate_qdrant_point_id(document.chunks[0].chunk_id)],
        with_vectors=True,
    )[0]

    assert written_count == document.chunk_count
    assert point.vector[DENSE_VECTOR_NAME] == dense_embeddings[0].vector
    assert point.vector[SPARSE_VECTOR_NAME].indices == sparse_embeddings[0].indices
    assert point.vector[SPARSE_VECTOR_NAME].values == sparse_embeddings[0].values


def test_hybrid_upsert_is_idempotent(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)
    dense_embeddings = create_dense_embeddings(document)
    sparse_embeddings = create_sparse_embeddings(document)

    sparse_vector_store.upsert_hybrid_document(
        ingested_document=document,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )
    sparse_vector_store.upsert_hybrid_document(
        ingested_document=document,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    results = sparse_vector_store.search_sparse(
        query_indices=sparse_embeddings[0].indices,
        query_values=sparse_embeddings[0].values,
        limit=20,
        document_id=document.document.document_id,
    )

    assert len(results) == document.chunk_count


def test_hybrid_upsert_rejects_count_mismatch(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)

    with pytest.raises(
        VectorStoreDataError,
        match="counts must match",
    ):
        sparse_vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=create_dense_embeddings(document),
            sparse_embeddings=create_sparse_embeddings(document)[:-1],
        )


def test_hybrid_upsert_rejects_dense_chunk_id_mismatch(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)
    dense_embeddings = create_dense_embeddings(document)
    dense_embeddings[0] = dense_embeddings[0].model_copy(
        update={"chunk_id": "a" * 64},
    )

    with pytest.raises(
        VectorStoreDataError,
        match="Embeddings do not correspond",
    ):
        sparse_vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=create_sparse_embeddings(document),
        )


def test_hybrid_upsert_rejects_sparse_chunk_id_mismatch(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)
    sparse_embeddings = create_sparse_embeddings(document)
    sparse_embeddings[0] = sparse_embeddings[0].model_copy(
        update={"chunk_id": "a" * 64},
    )

    with pytest.raises(
        VectorStoreDataError,
        match="Sparse embeddings do not correspond",
    ):
        sparse_vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=create_dense_embeddings(document),
            sparse_embeddings=sparse_embeddings,
        )


def test_hybrid_upsert_rejects_document_id_mismatch(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)
    sparse_embeddings = create_sparse_embeddings(document)
    sparse_embeddings[0] = sparse_embeddings[0].model_copy(
        update={"document_id": "a" * 64},
    )

    with pytest.raises(
        VectorStoreDataError,
        match="must belong to one document",
    ):
        sparse_vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=create_dense_embeddings(document),
            sparse_embeddings=sparse_embeddings,
        )


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


def test_sparse_search_retrieves_lexical_match(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(
        tmp_path,
        content="Remote device policy allows employees to work from home.",
    )
    sparse_provider = HashedLexicalSparseProvider()
    query_embedding = sparse_provider.embed_query("remote device")

    sparse_vector_store.upsert_hybrid_document(
        ingested_document=document,
        dense_embeddings=create_dense_embeddings(document),
        sparse_embeddings=sparse_provider.embed_chunks(document.chunks),
    )

    results = sparse_vector_store.search_sparse(
        query_indices=query_embedding.indices,
        query_values=query_embedding.values,
        limit=3,
    )

    assert results
    assert results[0].document_id == document.document.document_id
    assert "Remote device policy" in results[0].text


def test_sparse_search_supports_document_id_filtering(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    remote_document = create_ingested_document(
        tmp_path,
        file_name="remote.txt",
        content="Remote device policy allows employees to work from home.",
    )
    travel_document = create_ingested_document(
        tmp_path,
        file_name="travel.txt",
        content="Travel reimbursement policy covers hotels and meals.",
    )
    sparse_provider = HashedLexicalSparseProvider()

    for document in (remote_document, travel_document):
        sparse_vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=create_dense_embeddings(document),
            sparse_embeddings=sparse_provider.embed_chunks(document.chunks),
        )

    query_embedding = sparse_provider.embed_query("policy")

    results = sparse_vector_store.search_sparse(
        query_indices=query_embedding.indices,
        query_values=query_embedding.values,
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

    deleted_chunks = vector_store.delete_document(document.document.document_id)

    query_embedding = embedding_provider.embed_query("remote employee policy")

    results = vector_store.search_dense(
        query_vector=query_embedding.vector,
        limit=5,
        document_id=document.document.document_id,
    )

    assert deleted_chunks == document.chunk_count
    assert results == []


def test_list_documents_aggregates_chunks_without_text_or_vectors(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)
    sparse_vector_store.upsert_hybrid_document(
        ingested_document=document,
        dense_embeddings=create_dense_embeddings(document),
        sparse_embeddings=create_sparse_embeddings(document),
    )

    documents, next_cursor = sparse_vector_store.list_documents(limit=20)

    assert next_cursor is None
    assert len(documents) == 1
    assert documents[0].document_id == document.document.document_id
    assert documents[0].filename == document.document.file_name
    assert documents[0].content_hash == document.document.content_hash
    assert documents[0].chunk_count == document.chunk_count
    assert "text" not in documents[0].model_dump()
    assert "vector" not in documents[0].model_dump()


def test_get_document_deduplicates_and_sorts_metadata(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(
        tmp_path,
        content=(
            "Heading\nRemote work policy allows work from home. Remote work repeats."
        ),
    )
    sparse_vector_store.upsert_hybrid_document(
        ingested_document=document,
        dense_embeddings=create_dense_embeddings(document),
        sparse_embeddings=create_sparse_embeddings(document),
    )

    detail = sparse_vector_store.get_document(document.document.document_id)

    assert detail is not None
    assert detail.chunk_count == document.chunk_count
    assert detail.chunk_indices == sorted(
        {chunk.chunk_index for chunk in document.chunks}
    )
    assert detail.page_numbers == sorted(
        {chunk.page_number for chunk in document.chunks if chunk.page_number}
    )
    assert detail.headings == sorted(
        {chunk.heading for chunk in document.chunks if chunk.heading}
    )


def test_get_document_returns_none_for_missing_document(
    sparse_vector_store: QdrantVectorStore,
) -> None:
    assert sparse_vector_store.get_document("a" * 64) is None


def test_delete_document_returns_zero_for_missing_document(
    sparse_vector_store: QdrantVectorStore,
) -> None:
    assert sparse_vector_store.delete_document("a" * 64) == 0


def test_list_documents_uses_document_cursor(
    tmp_path: Path,
    sparse_vector_store: QdrantVectorStore,
) -> None:
    first = create_ingested_document(tmp_path, file_name="a.txt", content="alpha one")
    second = create_ingested_document(tmp_path, file_name="b.txt", content="beta two")
    for document in (first, second):
        sparse_vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=create_dense_embeddings(document),
            sparse_embeddings=create_sparse_embeddings(document),
        )

    documents, next_cursor = sparse_vector_store.list_documents(limit=1)
    next_documents, _ = sparse_vector_store.list_documents(
        limit=1,
        cursor=next_cursor,
    )

    assert next_cursor == documents[0].document_id
    assert next_documents
    assert next_documents[0].document_id > next_cursor


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


def test_hybrid_upsert_rejects_sparse_disabled(
    tmp_path: Path,
    vector_store: QdrantVectorStore,
) -> None:
    document = create_ingested_document(tmp_path)

    with pytest.raises(
        VectorStoreConfigurationError,
        match="requires sparse_enabled=True",
    ):
        vector_store.upsert_hybrid_document(
            ingested_document=document,
            dense_embeddings=create_dense_embeddings(document, dimensions=384),
            sparse_embeddings=create_sparse_embeddings(document),
        )


def test_sparse_search_rejects_sparse_disabled(
    vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(
        VectorStoreConfigurationError,
        match="requires sparse_enabled=True",
    ):
        vector_store.search_sparse(
            query_indices=[1],
            query_values=[1.0],
        )


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ([], [], "cannot be empty"),
        ([1, 2], [1.0], "equal lengths"),
        ([-1], [1.0], "non-negative"),
        ([1, 1], [1.0, 2.0], "unique"),
        ([2, 1], [1.0, 2.0], "sorted"),
        ([1], [math.inf], "finite"),
        ([1], [math.nan], "finite"),
    ],
)
def test_sparse_search_rejects_invalid_vectors(
    sparse_vector_store: QdrantVectorStore,
    indices: list[int],
    values: list[float],
    message: str,
) -> None:
    with pytest.raises(
        VectorStoreDataError,
        match=message,
    ):
        sparse_vector_store.search_sparse(
            query_indices=indices,
            query_values=values,
        )


def test_sparse_search_rejects_invalid_limit(
    sparse_vector_store: QdrantVectorStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        sparse_vector_store.search_sparse(
            query_indices=[1],
            query_values=[1.0],
            limit=0,
        )
