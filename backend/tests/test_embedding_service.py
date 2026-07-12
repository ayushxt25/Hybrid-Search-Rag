from pathlib import Path

import pytest

from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.pipeline import DocumentIngestionPipeline


@pytest.fixture(scope="module")
def embedding_provider() -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider()


@pytest.fixture(scope="module")
def sample_chunks(tmp_path_factory: pytest.TempPathFactory):
    temporary_directory = tmp_path_factory.mktemp("embedding-documents")
    document_path = Path(temporary_directory) / "policy.txt"

    document_path.write_text(
        (
            "Employees may work remotely for three days per week. "
            "Employees receive eighteen paid leave days."
        ),
        encoding="utf-8",
    )

    pipeline = DocumentIngestionPipeline(
        chunk_size=8,
        chunk_overlap=2,
    )

    return pipeline.ingest(document_path).chunks


def test_provider_reports_expected_dimensions(
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    assert embedding_provider.dimensions == 384


def test_provider_embeds_all_chunks(
    embedding_provider: SentenceTransformerEmbeddingProvider,
    sample_chunks,
) -> None:
    embeddings = embedding_provider.embed_chunks(sample_chunks)

    assert len(embeddings) == len(sample_chunks)

    for chunk, embedding in zip(
        sample_chunks,
        embeddings,
        strict=True,
    ):
        assert embedding.chunk_id == chunk.chunk_id
        assert embedding.document_id == chunk.document_id
        assert embedding.dimensions == 384
        assert len(embedding.vector) == 384


def test_provider_generates_normalized_query_embedding(
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    embedding = embedding_provider.embed_query("Can employees work from home?")

    squared_magnitude = sum(value * value for value in embedding.vector)

    assert embedding.query == "Can employees work from home?"
    assert embedding.dimensions == 384
    assert squared_magnitude == pytest.approx(1.0, abs=1e-5)


def test_provider_returns_empty_list_for_no_chunks(
    embedding_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    assert embedding_provider.embed_chunks([]) == []


@pytest.mark.parametrize(
    "query",
    ["", " ", "\n\t"],
)
def test_provider_rejects_empty_query(
    embedding_provider: SentenceTransformerEmbeddingProvider,
    query: str,
) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        embedding_provider.embed_query(query)


def test_provider_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        SentenceTransformerEmbeddingProvider(model_name=" ")
