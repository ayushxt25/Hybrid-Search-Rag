import math

import pytest

from app.schemas.document import TextChunk
from app.schemas.embedding import ChunkSparseEmbedding, QuerySparseEmbedding
from app.sparse.hashed_lexical import HashedLexicalSparseProvider

DOCUMENT_ID = "a" * 64
CHUNK_ID = "b" * 64
SECOND_CHUNK_ID = "c" * 64


def create_chunk(
    text: str,
    *,
    chunk_id: str = CHUNK_ID,
    document_id: str = DOCUMENT_ID,
) -> TextChunk:
    word_count = max(1, len(text.split()))

    return TextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        section_index=0,
        page_number=None,
        heading=None,
        text=text,
        start_word=0,
        end_word=word_count,
        word_count=word_count,
    )


def test_output_is_deterministic_across_provider_instances() -> None:
    chunk = create_chunk("Remote work remote policy")

    first = HashedLexicalSparseProvider().embed_chunks([chunk])[0]
    second = HashedLexicalSparseProvider().embed_chunks([chunk])[0]

    assert first == second


def test_same_token_maps_to_same_index() -> None:
    provider = HashedLexicalSparseProvider()

    assert provider.token_to_index("Remote") == provider.token_to_index("remote")


def test_document_and_query_token_indices_are_compatible() -> None:
    provider = HashedLexicalSparseProvider()

    query = provider.embed_query("Remote-work policy")
    chunk = provider.embed_chunks([create_chunk("remote-work policy!")])[0]

    assert query.indices == chunk.indices


def test_chunk_embedding_preserves_identity() -> None:
    provider = HashedLexicalSparseProvider()
    chunk = create_chunk(
        "Employees may work remotely.",
        chunk_id=SECOND_CHUNK_ID,
        document_id=DOCUMENT_ID,
    )

    embedding = provider.embed_chunks([chunk])[0]

    assert embedding.chunk_id == SECOND_CHUNK_ID
    assert embedding.document_id == DOCUMENT_ID


def test_empty_chunk_list_returns_empty_result() -> None:
    assert HashedLexicalSparseProvider().embed_chunks([]) == []


@pytest.mark.parametrize(
    "query",
    ["", " ", "\n\t"],
)
def test_whitespace_only_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        HashedLexicalSparseProvider().embed_query(query)


def test_chunk_with_no_usable_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunk text contains no usable tokens"):
        HashedLexicalSparseProvider().embed_chunks([create_chunk("... !!!")])


def test_query_with_no_usable_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="query contains no usable tokens"):
        HashedLexicalSparseProvider().embed_query("... !!!")


def test_punctuation_normalization() -> None:
    provider = HashedLexicalSparseProvider()

    first = provider.embed_query("remote, policy!")
    second = provider.embed_query("remote policy")

    assert first.indices == second.indices
    assert first.values == second.values


def test_case_normalization() -> None:
    provider = HashedLexicalSparseProvider()

    first = provider.embed_query("REMOTE Policy")
    second = provider.embed_query("remote policy")

    assert first.indices == second.indices
    assert first.values == second.values


def test_repeated_terms_use_saturated_weighting() -> None:
    provider = HashedLexicalSparseProvider(k1=1.2)

    single = provider.embed_query("remote")
    repeated = provider.embed_query("remote remote remote")

    assert single.indices == repeated.indices
    assert repeated.values[0] > single.values[0]
    assert repeated.values[0] < single.values[0] * 3


def test_indices_are_sorted_and_unique() -> None:
    embedding = HashedLexicalSparseProvider().embed_query("remote policy device")

    assert embedding.indices == sorted(embedding.indices)
    assert len(embedding.indices) == len(set(embedding.indices))


def test_indices_and_values_have_equal_lengths() -> None:
    embedding = HashedLexicalSparseProvider().embed_query("remote remote policy")

    assert len(embedding.indices) == len(embedding.values)


def test_indices_are_non_negative() -> None:
    embedding = HashedLexicalSparseProvider().embed_query("remote policy")

    assert all(index >= 0 for index in embedding.indices)


def test_values_are_finite() -> None:
    embedding = HashedLexicalSparseProvider().embed_query("remote remote policy")

    assert all(math.isfinite(value) for value in embedding.values)


def test_collision_accumulation_with_tiny_feature_space() -> None:
    provider = HashedLexicalSparseProvider(feature_space_size=1)

    embedding = provider.embed_query("remote policy")
    single_remote = provider.embed_query("remote")
    single_policy = provider.embed_query("policy")

    assert embedding.indices == [0]
    assert embedding.values == pytest.approx(
        [single_remote.values[0] + single_policy.values[0]]
    )


def test_schema_rejects_mismatched_sparse_lengths() -> None:
    with pytest.raises(ValueError, match="indices and values"):
        QuerySparseEmbedding(query="remote", indices=[1, 2], values=[1.0])


def test_schema_rejects_negative_sparse_indices() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        QuerySparseEmbedding(query="remote", indices=[-1], values=[1.0])


def test_schema_rejects_duplicate_sparse_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        QuerySparseEmbedding(query="remote", indices=[1, 1], values=[1.0, 2.0])


def test_schema_rejects_unsorted_sparse_indices() -> None:
    with pytest.raises(ValueError, match="sorted"):
        QuerySparseEmbedding(query="remote", indices=[2, 1], values=[1.0, 2.0])


def test_schema_rejects_non_finite_sparse_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ChunkSparseEmbedding(
            chunk_id=CHUNK_ID,
            document_id=DOCUMENT_ID,
            indices=[1],
            values=[math.inf],
        )


def test_schema_rejects_empty_sparse_vectors() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        QuerySparseEmbedding(query="remote", indices=[], values=[])


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="k1"):
        HashedLexicalSparseProvider(k1=0)

    with pytest.raises(ValueError, match="feature_space_size"):
        HashedLexicalSparseProvider(feature_space_size=0)
