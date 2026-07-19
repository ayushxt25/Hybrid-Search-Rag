import math
import time
from collections.abc import Callable
from typing import Any

from app.embeddings.base import DenseEmbeddingProvider
from app.schemas.document import TextChunk
from app.schemas.embedding import ChunkEmbedding, QueryEmbedding

RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


class GeminiEmbeddingError(RuntimeError):
    """Raised when Gemini embeddings cannot be generated safely."""


class GeminiEmbeddingProvider(DenseEmbeddingProvider):
    """Generate dense vectors with the Gemini Developer API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        dimensions: int = 768,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key cannot be empty.")
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.api_key = api_key
        self.model_name = model_name
        self._dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client
        self._sleep = sleep

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": int(self.timeout_seconds * 1000)},
            )
        return self._client

    def embed_chunks(self, chunks: list[TextChunk]) -> list[ChunkEmbedding]:
        if not chunks:
            return []

        vectors = self._embed_texts(
            [chunk.text for chunk in chunks],
            task_type=RETRIEVAL_DOCUMENT,
        )
        if len(vectors) != len(chunks):
            raise GeminiEmbeddingError("Gemini returned an unexpected vector count.")

        return [
            ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                vector=vector,
                dimensions=self.dimensions,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_query(self, query: str) -> QueryEmbedding:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty.")

        vectors = self._embed_texts([normalized_query], task_type=RETRIEVAL_QUERY)
        if len(vectors) != 1:
            raise GeminiEmbeddingError(
                "Gemini returned an unexpected query vector count."
            )

        return QueryEmbedding(
            query=normalized_query,
            vector=vectors[0],
            dimensions=self.dimensions,
        )

    def _embed_texts(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Gemini embedding text cannot be empty.")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._extract_vectors(
                    self.client.models.embed_content(
                        model=self.model_name,
                        contents=texts,
                        config=self._embed_config(task_type),
                    )
                )
            except Exception as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                self._sleep(0.25 * (2**attempt))

        raise GeminiEmbeddingError("Gemini embedding request failed.") from last_error

    def _embed_config(self, task_type: str):
        try:
            from google.genai import types

            return types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions,
            )
        except Exception:
            return {
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
            }

    def _extract_vectors(self, response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            embedding = getattr(response, "embedding", None)
            embeddings = [] if embedding is None else [embedding]
        if not isinstance(embeddings, list) or not embeddings:
            raise GeminiEmbeddingError("Gemini returned no embeddings.")

        vectors = [
            self._coerce_vector(getattr(item, "values", None)) for item in embeddings
        ]
        return [self._normalize(vector) for vector in vectors]

    def _coerce_vector(self, values: Any) -> list[float]:
        if not isinstance(values, list) or len(values) != self.dimensions:
            raise GeminiEmbeddingError("Gemini embedding dimensions are invalid.")
        vector = [float(value) for value in values]
        if any(not math.isfinite(value) for value in vector):
            raise GeminiEmbeddingError("Gemini embedding contains invalid values.")
        return vector

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude <= 0:
            raise GeminiEmbeddingError("Gemini embedding has zero magnitude.")
        return [value / magnitude for value in vector]
