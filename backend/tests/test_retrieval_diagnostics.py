import pytest
from pydantic import ValidationError

from app.retrieval.rrf import reciprocal_rank_fusion
from app.schemas.search import (
    BranchScoreDiagnostic,
    DenseSearchResult,
    RetrievalScoreDiagnostic,
)


def create_result(
    chunk_id: str,
    *,
    score: float,
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
        text="Safe result text.",
        start_word=0,
        end_word=3,
        word_count=3,
    )


def test_diagnostic_model_validation() -> None:
    diagnostic = RetrievalScoreDiagnostic(
        dense=BranchScoreDiagnostic(
            raw_score=0.9,
            rank=1,
            weight=1.5,
            rrf_contribution=1.5 / 61,
        ),
        sparse=BranchScoreDiagnostic(
            raw_score=None,
            rank=None,
            weight=1.0,
            rrf_contribution=0.0,
        ),
        fused_score=1.5 / 61,
        fused_rank=1,
    )

    assert diagnostic.dense.rank == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raw_score": float("nan"), "rank": 1, "weight": 1.0, "rrf_contribution": 0.0},
        {"raw_score": 0.1, "rank": 0, "weight": 1.0, "rrf_contribution": 0.0},
        {"raw_score": 0.1, "rank": 1, "weight": 0.0, "rrf_contribution": 0.0},
        {"raw_score": 0.1, "rank": 1, "weight": 1.0, "rrf_contribution": -0.1},
    ],
)
def test_branch_diagnostic_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        BranchScoreDiagnostic(**kwargs)


def test_hybrid_diagnostics_for_dense_only_result() -> None:
    result = create_result("a" * 64, score=0.9)

    fused = reciprocal_rank_fusion(
        [[result], []],
        limit=1,
        k=60,
        weights=[1.5, 1.0],
        include_score_diagnostics=True,
    )[0]

    diagnostic = fused.score_diagnostics
    assert diagnostic is not None
    assert diagnostic.dense.raw_score == 0.9
    assert diagnostic.dense.rank == 1
    assert diagnostic.dense.rrf_contribution == pytest.approx(1.5 / 61)
    assert diagnostic.sparse.raw_score is None
    assert diagnostic.sparse.rank is None
    assert diagnostic.sparse.weight == 1.0
    assert diagnostic.sparse.rrf_contribution == 0.0
    assert diagnostic.fused_score == pytest.approx(1.5 / 61)
    assert diagnostic.fused_rank == 1


def test_hybrid_diagnostics_for_sparse_only_result() -> None:
    result = create_result("a" * 64, score=0.7)

    fused = reciprocal_rank_fusion(
        [[], [result]],
        limit=1,
        k=60,
        weights=[1.5, 1.0],
        include_score_diagnostics=True,
    )[0]

    diagnostic = fused.score_diagnostics
    assert diagnostic is not None
    assert diagnostic.dense.raw_score is None
    assert diagnostic.dense.weight == 1.5
    assert diagnostic.dense.rrf_contribution == 0.0
    assert diagnostic.sparse.raw_score == 0.7
    assert diagnostic.sparse.rank == 1
    assert diagnostic.sparse.rrf_contribution == pytest.approx(1.0 / 61)


def test_hybrid_diagnostics_for_result_in_both_branches() -> None:
    result = create_result("a" * 64, score=0.9)
    sparse_result = result.model_copy(update={"score": 0.4})

    fused = reciprocal_rank_fusion(
        [[create_result("b" * 64, score=0.8), result], [sparse_result]],
        limit=2,
        k=60,
        weights=[1.5, 1.0],
        include_score_diagnostics=True,
    )

    duplicate = next(item for item in fused if item.chunk_id == result.chunk_id)
    diagnostic = duplicate.score_diagnostics
    assert diagnostic is not None
    assert diagnostic.dense.rank == 2
    assert diagnostic.sparse.rank == 1
    assert diagnostic.dense.raw_score == 0.9
    assert diagnostic.sparse.raw_score == 0.4
    assert diagnostic.dense.rrf_contribution == pytest.approx(1.5 / 62)
    assert diagnostic.sparse.rrf_contribution == pytest.approx(1.0 / 61)
    assert diagnostic.fused_score == pytest.approx(
        diagnostic.dense.rrf_contribution + diagnostic.sparse.rrf_contribution
    )
    assert diagnostic.fused_rank == fused.index(duplicate) + 1


def test_diagnostics_do_not_contain_vectors_or_raw_payloads() -> None:
    result = create_result("a" * 64, score=0.9)

    fused = reciprocal_rank_fusion(
        [[result], []],
        limit=1,
        include_score_diagnostics=True,
    )[0]

    dumped = fused.score_diagnostics.model_dump()
    assert "vector" not in dumped
    assert "indices" not in dumped
    assert "payload" not in dumped
    assert "point_id" not in dumped
