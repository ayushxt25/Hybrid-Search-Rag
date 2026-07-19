from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.embeddings.gemini import (
    RETRIEVAL_DOCUMENT,
    RETRIEVAL_QUERY,
    GeminiEmbeddingError,
    GeminiEmbeddingProvider,
)
from app.schemas.document import TextChunk


def chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id="d" * 64,
        text=text,
        chunk_index=0,
        section_index=0,
        start_word=0,
        end_word=1,
        word_count=1,
        page_number=None,
        heading=None,
    )


def response(*vectors: list[float]):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors]
    )


def config_value(config, snake_name: str, camel_name: str):
    if isinstance(config, dict):
        return config[snake_name]
    return getattr(config, snake_name, getattr(config, camel_name, None))


def provider_with_response(result):
    client = SimpleNamespace(models=Mock())
    client.models.embed_content.return_value = result
    provider = GeminiEmbeddingProvider(
        api_key="secret",
        dimensions=3,
        client=client,
        sleep=lambda _: None,
    )
    return provider, client


def test_gemini_embeds_documents_with_retrieval_document_task() -> None:
    provider, client = provider_with_response(
        response([1.0, 0.0, 0.0], [0.0, 2.0, 0.0])
    )
    chunks = [chunk("a" * 64, "alpha"), chunk("b" * 64, "beta")]

    embeddings = provider.embed_chunks(chunks)

    assert [embedding.chunk_id for embedding in embeddings] == ["a" * 64, "b" * 64]
    assert embeddings[1].vector == [0.0, 1.0, 0.0]
    call = client.models.embed_content.call_args.kwargs
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == ["alpha", "beta"]
    assert config_value(call["config"], "task_type", "taskType") == RETRIEVAL_DOCUMENT
    assert (
        config_value(call["config"], "output_dimensionality", "outputDimensionality")
        == 3
    )


def test_gemini_embeds_query_with_retrieval_query_task() -> None:
    provider, client = provider_with_response(response([0.0, 3.0, 0.0]))

    embedding = provider.embed_query(" remote policy ")

    assert embedding.query == "remote policy"
    assert embedding.dimensions == 3
    assert embedding.vector == [0.0, 1.0, 0.0]
    call = client.models.embed_content.call_args.kwargs
    assert call["contents"] == ["remote policy"]
    assert config_value(call["config"], "task_type", "taskType") == RETRIEVAL_QUERY


def test_gemini_rejects_malformed_response() -> None:
    provider, _ = provider_with_response(response([1.0, 2.0]))

    with pytest.raises(GeminiEmbeddingError, match="request failed"):
        provider.embed_query("policy")


def test_gemini_maps_provider_failure_after_bounded_retries() -> None:
    client = SimpleNamespace(models=Mock())
    client.models.embed_content.side_effect = RuntimeError("raw provider detail")
    provider = GeminiEmbeddingProvider(
        api_key="secret",
        dimensions=3,
        client=client,
        max_retries=1,
        sleep=lambda _: None,
    )

    with pytest.raises(GeminiEmbeddingError, match="request failed") as error:
        provider.embed_query("policy")

    assert "raw provider detail" not in str(error.value)
    assert client.models.embed_content.call_count == 2


def test_gemini_import_does_not_create_client() -> None:
    provider = GeminiEmbeddingProvider(api_key="secret", dimensions=3)

    assert provider._client is None
