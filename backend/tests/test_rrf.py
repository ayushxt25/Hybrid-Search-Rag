import pytest

from app.retrieval.rrf import reciprocal_rank_fusion
from app.schemas.search import DenseSearchResult


def create_result(
    chunk_id: str,
    *,
    score: float = 0.5,
) -> DenseSearchResult:
    return DenseSearchResult(
        point_id=f"point-{chunk_id}",
        chunk_id=chunk_id,
        document_id="d" * 64,
        score=score,
        file_name="policy.txt",
        file_extension=".txt",
        chunk_index=0,
        section_index=0,
        page_number=None,
        heading=None,
        text=f"Chunk {chunk_id}",
        start_word=0,
        end_word=2,
        word_count=2,
    )


def test_rrf_combines_two_ranked_lists() -> None:
    first = create_result("a" * 64)
    duplicate = create_result("b" * 64)
    unique = create_result("c" * 64)

    results = reciprocal_rank_fusion(
        [
            [first, duplicate],
            [duplicate, unique],
        ],
        limit=3,
        k=60,
    )

    assert [result.chunk_id for result in results] == [
        duplicate.chunk_id,
        first.chunk_id,
        unique.chunk_id,
    ]
    assert results[0].score == pytest.approx((1 / 62) + (1 / 61))
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)


def test_rrf_enforces_limit() -> None:
    results = reciprocal_rank_fusion(
        [[create_result("a" * 64), create_result("b" * 64)]],
        limit=1,
    )

    assert len(results) == 1


def test_rrf_empty_input_returns_empty_result() -> None:
    assert reciprocal_rank_fusion([], limit=5) == []
    assert reciprocal_rank_fusion([[]], limit=5) == []


def test_rrf_supports_one_empty_list() -> None:
    result = create_result("a" * 64)

    results = reciprocal_rank_fusion(
        [
            [],
            [result],
        ],
        limit=5,
    )

    assert [fused.chunk_id for fused in results] == [result.chunk_id]


def test_rrf_does_not_mutate_input_results() -> None:
    result = create_result("a" * 64, score=0.75)

    fused = reciprocal_rank_fusion([[result]], limit=1)

    assert result.score == 0.75
    assert fused[0] is not result
    assert fused[0].score != result.score


def test_rrf_uses_deterministic_tie_breaking() -> None:
    later_rank = create_result("a" * 64)
    lower_chunk_id = create_result("b" * 64)
    higher_chunk_id = create_result("c" * 64)

    results = reciprocal_rank_fusion(
        [
            [later_rank],
            [lower_chunk_id, higher_chunk_id],
            [higher_chunk_id, lower_chunk_id],
        ],
        limit=3,
        k=60,
    )

    assert [result.chunk_id for result in results] == [
        lower_chunk_id.chunk_id,
        higher_chunk_id.chunk_id,
        later_rank.chunk_id,
    ]


def test_rrf_rejects_invalid_limit() -> None:
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        reciprocal_rank_fusion([], limit=0)


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        reciprocal_rank_fusion([], limit=1, k=0)
