import math
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.schemas.document import IngestedDocument, TextChunk
from app.schemas.embedding import ChunkEmbedding, ChunkSparseEmbedding
from app.schemas.search import DenseSearchResult
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)
from app.vectorstore.identifiers import generate_qdrant_point_id

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore:
    """Store and retrieve document-chunk embeddings using Qdrant."""

    def __init__(
        self,
        *,
        collection_name: str,
        vector_dimensions: int,
        url: str | None = None,
        client: QdrantClient | None = None,
        sparse_enabled: bool = False,
    ) -> None:
        normalized_collection_name = collection_name.strip()

        if not normalized_collection_name:
            raise VectorStoreConfigurationError("collection_name cannot be empty.")

        if vector_dimensions <= 0:
            raise VectorStoreConfigurationError(
                "vector_dimensions must be greater than zero."
            )

        if client is None and (url is None or not url.strip()):
            raise VectorStoreConfigurationError(
                "Either a Qdrant client or URL must be provided."
            )

        self.collection_name = normalized_collection_name
        self.vector_dimensions = vector_dimensions
        self._owns_client = client is None
        self._closed = False
        self.client = client if client is not None else QdrantClient(url=url)
        self.sparse_enabled = sparse_enabled

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._owns_client:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

    def ensure_collection(self) -> None:
        """Create the configured collection when it does not exist."""
        try:
            if self.client.collection_exists(self.collection_name):
                self._validate_existing_collection()
                return

            collection_config: dict[str, Any] = {
                "collection_name": self.collection_name,
                "vectors_config": {
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.vector_dimensions,
                        distance=models.Distance.COSINE,
                    )
                },
            }

            if self.sparse_enabled:
                collection_config["sparse_vectors_config"] = {
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(),
                }

            self.client.create_collection(**collection_config)
        except VectorStoreConfigurationError:
            raise
        except (
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            raise VectorStoreConnectionError(
                "Unable to create or inspect the Qdrant collection."
            ) from error

    def _validate_existing_collection(self) -> None:
        """Validate an existing collection's dense-vector configuration."""
        collection = self.client.get_collection(self.collection_name)
        vectors_config = collection.config.params.vectors

        if not isinstance(vectors_config, dict):
            raise VectorStoreConfigurationError(
                "Qdrant collection does not use named vectors."
            )

        dense_config = vectors_config.get(DENSE_VECTOR_NAME)

        if dense_config is None:
            raise VectorStoreConfigurationError(
                "Qdrant collection is missing the named dense vector."
            )

        if dense_config.size != self.vector_dimensions:
            raise VectorStoreConfigurationError(
                "Qdrant dense-vector dimensions do not match the embedding provider."
            )

        if dense_config.distance != models.Distance.COSINE:
            raise VectorStoreConfigurationError(
                "Qdrant dense vector must use cosine distance."
            )

        if not self.sparse_enabled:
            return

        sparse_vectors_config = collection.config.params.sparse_vectors

        if (
            not isinstance(sparse_vectors_config, dict)
            or SPARSE_VECTOR_NAME not in sparse_vectors_config
        ):
            raise VectorStoreConfigurationError(
                "Qdrant collection is missing the named sparse vector."
            )

    def upsert_document(
        self,
        *,
        ingested_document: IngestedDocument,
        embeddings: Sequence[ChunkEmbedding],
    ) -> int:
        """Insert or update every chunk belonging to one document."""
        chunks = ingested_document.chunks

        self._validate_chunk_embeddings(
            ingested_document=ingested_document,
            chunks=chunks,
            embeddings=embeddings,
        )

        embedding_by_chunk_id = {
            embedding.chunk_id: embedding for embedding in embeddings
        }

        points: list[models.PointStruct] = []

        for chunk in chunks:
            embedding = embedding_by_chunk_id[chunk.chunk_id]

            points.append(
                models.PointStruct(
                    id=generate_qdrant_point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: embedding.vector,
                    },
                    payload=self._build_payload(
                        ingested_document=ingested_document,
                        chunk=chunk,
                    ),
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except (
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            raise VectorStoreConnectionError(
                "Unable to write document chunks to Qdrant."
            ) from error

        return len(points)

    def upsert_hybrid_document(
        self,
        *,
        ingested_document: IngestedDocument,
        dense_embeddings: Sequence[ChunkEmbedding],
        sparse_embeddings: Sequence[ChunkSparseEmbedding],
    ) -> int:
        """Insert or update dense and sparse vectors for every document chunk."""
        self._require_sparse_enabled("Hybrid document upsert")

        chunks = ingested_document.chunks

        self._validate_hybrid_chunk_embeddings(
            ingested_document=ingested_document,
            chunks=chunks,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        dense_embedding_by_chunk_id = {
            embedding.chunk_id: embedding for embedding in dense_embeddings
        }
        sparse_embedding_by_chunk_id = {
            embedding.chunk_id: embedding for embedding in sparse_embeddings
        }

        points: list[models.PointStruct] = []

        for chunk in chunks:
            dense_embedding = dense_embedding_by_chunk_id[chunk.chunk_id]
            sparse_embedding = sparse_embedding_by_chunk_id[chunk.chunk_id]

            points.append(
                models.PointStruct(
                    id=generate_qdrant_point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_embedding.vector,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_embedding.indices,
                            values=sparse_embedding.values,
                        ),
                    },
                    payload=self._build_payload(
                        ingested_document=ingested_document,
                        chunk=chunk,
                    ),
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except (
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            raise VectorStoreConnectionError(
                "Unable to write hybrid document chunks to Qdrant."
            ) from error

        return len(points)

    def _validate_chunk_embeddings(
        self,
        *,
        ingested_document: IngestedDocument,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[ChunkEmbedding],
    ) -> None:
        """Validate consistency between the document, chunks, and embeddings."""
        if not chunks:
            raise VectorStoreDataError("At least one document chunk is required.")

        if len(chunks) != len(embeddings):
            raise VectorStoreDataError("Chunk and embedding counts must match.")

        chunk_ids = {chunk.chunk_id for chunk in chunks}
        embedding_chunk_ids = {embedding.chunk_id for embedding in embeddings}

        if len(chunk_ids) != len(chunks):
            raise VectorStoreDataError("Document contains duplicate chunk identifiers.")

        if len(embedding_chunk_ids) != len(embeddings):
            raise VectorStoreDataError(
                "Embeddings contain duplicate chunk identifiers."
            )

        if chunk_ids != embedding_chunk_ids:
            raise VectorStoreDataError(
                "Embeddings do not correspond to the supplied chunks."
            )

        document_ids = {chunk.document_id for chunk in chunks}

        if len(document_ids) != 1:
            raise VectorStoreDataError("Chunks must belong to one document.")

        expected_document_id = next(iter(document_ids))
        parent_document_id = ingested_document.document.document_id

        if expected_document_id != parent_document_id:
            raise VectorStoreDataError(
                "Chunk document IDs do not match the ingested document."
            )

        for embedding in embeddings:
            if embedding.document_id != expected_document_id:
                raise VectorStoreDataError(
                    "All chunks and embeddings must belong to one document."
                )

            if embedding.dimensions != self.vector_dimensions:
                raise VectorStoreDataError(
                    "Embedding dimensions do not match the collection."
                )

            if len(embedding.vector) != self.vector_dimensions:
                raise VectorStoreDataError(
                    "Embedding vector length does not match its dimensions."
                )

    def _validate_hybrid_chunk_embeddings(
        self,
        *,
        ingested_document: IngestedDocument,
        chunks: Sequence[TextChunk],
        dense_embeddings: Sequence[ChunkEmbedding],
        sparse_embeddings: Sequence[ChunkSparseEmbedding],
    ) -> None:
        """Validate consistency between document chunks and hybrid embeddings."""
        self._validate_chunk_embeddings(
            ingested_document=ingested_document,
            chunks=chunks,
            embeddings=dense_embeddings,
        )

        if len(chunks) != len(sparse_embeddings):
            raise VectorStoreDataError(
                "Chunk, dense embedding, and sparse embedding counts must match."
            )

        chunk_ids = {chunk.chunk_id for chunk in chunks}
        sparse_embedding_chunk_ids = {
            embedding.chunk_id for embedding in sparse_embeddings
        }

        if len(sparse_embedding_chunk_ids) != len(sparse_embeddings):
            raise VectorStoreDataError(
                "Sparse embeddings contain duplicate chunk identifiers."
            )

        if chunk_ids != sparse_embedding_chunk_ids:
            raise VectorStoreDataError(
                "Sparse embeddings do not correspond to the supplied chunks."
            )

        expected_document_id = ingested_document.document.document_id

        for embedding in sparse_embeddings:
            if embedding.document_id != expected_document_id:
                raise VectorStoreDataError(
                    "All chunks and embeddings must belong to one document."
                )

            self._validate_sparse_vector(
                indices=embedding.indices,
                values=embedding.values,
            )

    @staticmethod
    def _build_payload(
        *,
        ingested_document: IngestedDocument,
        chunk: TextChunk,
    ) -> dict[str, Any]:
        """Build searchable metadata for one Qdrant point."""
        document = ingested_document.document

        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content_hash": document.content_hash,
            "file_name": document.file_name,
            "file_extension": document.file_extension,
            "chunk_index": chunk.chunk_index,
            "section_index": chunk.section_index,
            "page_number": chunk.page_number,
            "heading": chunk.heading,
            "text": chunk.text,
            "start_word": chunk.start_word,
            "end_word": chunk.end_word,
            "word_count": chunk.word_count,
        }

    def search_dense(
        self,
        *,
        query_vector: Sequence[float],
        limit: int = 5,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> list[DenseSearchResult]:
        """Return chunks nearest to a dense query vector."""
        if len(query_vector) != self.vector_dimensions:
            raise VectorStoreDataError(
                "Query vector dimensions do not match the collection."
            )

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_filter = self._build_document_filter(document_id)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
        except (
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            raise VectorStoreConnectionError(
                "Unable to search the Qdrant collection."
            ) from error

        return [self._convert_search_result(point) for point in response.points]

    def search_sparse(
        self,
        *,
        query_indices: Sequence[int],
        query_values: Sequence[float],
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[DenseSearchResult]:
        """Return chunks nearest to a sparse lexical query vector."""
        self._require_sparse_enabled("Sparse search")

        self._validate_sparse_vector(
            indices=query_indices,
            values=query_values,
        )

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_filter = self._build_document_filter(document_id)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(
                    indices=list(query_indices),
                    values=list(query_values),
                ),
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except (
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            raise VectorStoreConnectionError(
                "Unable to search the Qdrant collection."
            ) from error

        return [self._convert_search_result(point) for point in response.points]

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """Delete every stored point belonging to one document."""
        normalized_document_id = document_id.strip()

        if not normalized_document_id:
            raise ValueError("document_id cannot be empty.")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(
                                    value=normalized_document_id,
                                ),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except (
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            raise VectorStoreConnectionError(
                "Unable to delete document vectors from Qdrant."
            ) from error

    @staticmethod
    def _build_document_filter(
        document_id: str | None,
    ) -> models.Filter | None:
        """Build an optional Qdrant document filter."""
        if document_id is None:
            return None

        normalized_document_id = document_id.strip()

        if not normalized_document_id:
            raise ValueError("document_id cannot be empty.")

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=normalized_document_id,
                    ),
                )
            ]
        )

    def _require_sparse_enabled(
        self,
        operation_name: str,
    ) -> None:
        if not self.sparse_enabled:
            raise VectorStoreConfigurationError(
                f"{operation_name} requires sparse_enabled=True."
            )

    @staticmethod
    def _validate_sparse_vector(
        *,
        indices: Sequence[int],
        values: Sequence[float],
    ) -> None:
        if len(indices) != len(values):
            raise VectorStoreDataError(
                "Sparse vector indices and values must have equal lengths."
            )

        if not indices:
            raise VectorStoreDataError("Sparse vectors cannot be empty.")

        if any(index < 0 for index in indices):
            raise VectorStoreDataError("Sparse vector indices must be non-negative.")

        if len(set(indices)) != len(indices):
            raise VectorStoreDataError("Sparse vector indices must be unique.")

        if list(indices) != sorted(indices):
            raise VectorStoreDataError(
                "Sparse vector indices must be sorted in ascending order."
            )

        if any(not math.isfinite(value) for value in values):
            raise VectorStoreDataError(
                "Sparse vector values must contain only finite numbers."
            )

    def _convert_search_result(
        self,
        point: models.ScoredPoint,
    ) -> DenseSearchResult:
        """Convert one Qdrant result into an application schema."""
        payload = point.payload or {}

        return DenseSearchResult(
            point_id=str(point.id),
            chunk_id=self._require_payload_string(
                payload,
                "chunk_id",
            ),
            document_id=self._require_payload_string(
                payload,
                "document_id",
            ),
            score=point.score,
            file_name=self._require_payload_string(
                payload,
                "file_name",
            ),
            file_extension=self._require_payload_string(
                payload,
                "file_extension",
            ),
            chunk_index=self._require_payload_int(
                payload,
                "chunk_index",
            ),
            section_index=self._require_payload_int(
                payload,
                "section_index",
            ),
            page_number=self._optional_payload_int(
                payload,
                "page_number",
            ),
            heading=self._optional_payload_string(
                payload,
                "heading",
            ),
            text=self._require_payload_string(
                payload,
                "text",
            ),
            start_word=self._require_payload_int(
                payload,
                "start_word",
            ),
            end_word=self._require_payload_int(
                payload,
                "end_word",
            ),
            word_count=self._require_payload_int(
                payload,
                "word_count",
            ),
        )

    @staticmethod
    def _require_payload_string(
        payload: dict[str, Any],
        key: str,
    ) -> str:
        value = payload.get(key)

        if not isinstance(value, str) or not value:
            raise VectorStoreDataError(
                f"Qdrant payload field '{key}' is missing or invalid."
            )

        return value

    @staticmethod
    def _optional_payload_string(
        payload: dict[str, Any],
        key: str,
    ) -> str | None:
        value = payload.get(key)

        if value is None:
            return None

        if not isinstance(value, str):
            raise VectorStoreDataError(f"Qdrant payload field '{key}' is invalid.")

        return value

    @staticmethod
    def _require_payload_int(
        payload: dict[str, Any],
        key: str,
    ) -> int:
        value = payload.get(key)

        if not isinstance(value, int) or isinstance(
            value,
            bool,
        ):
            raise VectorStoreDataError(
                f"Qdrant payload field '{key}' is missing or invalid."
            )

        return value

    @staticmethod
    def _optional_payload_int(
        payload: dict[str, Any],
        key: str,
    ) -> int | None:
        value = payload.get(key)

        if value is None:
            return None

        if not isinstance(value, int) or isinstance(
            value,
            bool,
        ):
            raise VectorStoreDataError(f"Qdrant payload field '{key}' is invalid.")

        return value
