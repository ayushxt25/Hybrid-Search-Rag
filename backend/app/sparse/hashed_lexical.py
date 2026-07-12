from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from app.schemas.document import TextChunk
from app.schemas.embedding import ChunkSparseEmbedding, QuerySparseEmbedding
from app.sparse.base import SparseEmbeddingProvider

DEFAULT_FEATURE_SPACE_SIZE = 2**20
TOKEN_PATTERN = re.compile(r"[a-z0-9_][a-z0-9_-]*")


@dataclass(frozen=True)
class SparseValues:
    indices: list[int]
    values: list[float]


class HashedLexicalSparseProvider(SparseEmbeddingProvider):
    """Generate deterministic hashed lexical sparse vectors.

    This is not corpus-fitted BM25. It uses stable hashed token indices and a
    BM25-style saturated term-frequency weight:
    ((k1 + 1) * frequency) / (k1 + frequency).
    """

    def __init__(
        self,
        *,
        feature_space_size: int = DEFAULT_FEATURE_SPACE_SIZE,
        k1: float = 1.2,
    ) -> None:
        if feature_space_size <= 0:
            raise ValueError("feature_space_size must be greater than zero.")

        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")

        self.feature_space_size = feature_space_size
        self.k1 = k1

    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[ChunkSparseEmbedding]:
        """Generate sparse lexical vectors for document chunks."""
        if not chunks:
            return []

        embeddings: list[ChunkSparseEmbedding] = []

        for chunk in chunks:
            sparse_values = self._encode_text(
                chunk.text,
                empty_text_message="chunk text contains no usable tokens.",
            )
            embeddings.append(
                ChunkSparseEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    indices=sparse_values.indices,
                    values=sparse_values.values,
                )
            )

        return embeddings

    def embed_query(self, query: str) -> QuerySparseEmbedding:
        """Generate a sparse lexical vector for a non-empty query."""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty.")

        sparse_values = self._encode_text(
            normalized_query,
            empty_text_message="query contains no usable tokens.",
        )

        return QuerySparseEmbedding(
            query=normalized_query,
            indices=sparse_values.indices,
            values=sparse_values.values,
        )

    def tokenize(self, text: str) -> list[str]:
        """Normalize text into tokens shared by document and query encoding."""
        return TOKEN_PATTERN.findall(text.lower())

    def token_to_index(self, token: str) -> int:
        """Map a token to a deterministic non-negative sparse index."""
        normalized_token = token.strip().lower()

        if not normalized_token:
            raise ValueError("token cannot be empty.")

        digest = hashlib.blake2b(
            normalized_token.encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, byteorder="big") % self.feature_space_size

    def _encode_text(
        self,
        text: str,
        *,
        empty_text_message: str,
    ) -> SparseValues:
        token_counts = Counter(self.tokenize(text))

        if not token_counts:
            raise ValueError(empty_text_message)

        values_by_index: dict[int, float] = {}

        for token, frequency in token_counts.items():
            index = self.token_to_index(token)
            weight = self._term_frequency_weight(frequency)
            values_by_index[index] = values_by_index.get(index, 0.0) + weight

        indices = sorted(values_by_index)
        values = [values_by_index[index] for index in indices]

        return SparseValues(indices=indices, values=values)

    def _term_frequency_weight(self, frequency: int) -> float:
        if frequency <= 0:
            raise ValueError("frequency must be greater than zero.")

        return ((self.k1 + 1.0) * frequency) / (self.k1 + frequency)
