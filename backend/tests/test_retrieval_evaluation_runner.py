from unittest.mock import Mock

import pytest

from app.evaluation.models import RetrievalEvaluationCase
from app.evaluation.runner import RetrievalEvaluationRunner
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import (
    DenseSearchResponse,
    HybridSearchResponse,
    SparseSearchResponse,
)

DOCUMENT_ID = "d" * 64
RELEVANT = "a" * 64
IRRELEVANT = "b" * 64


def create_result(chunk_id: str) -> DenseSearchResult:
    return DenseSearchResult(
        point_id=f"point-{chunk_id}",
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        score=0.5,
        file_name="policy.txt",
        file_extension=".txt",
        chunk_index=0,
        section_index=0,
        page_number=None,
        heading=None,
        text="Policy chunk.",
        start_word=0,
        end_word=2,
        word_count=2,
    )


def create_case(
    case_id: str = "remote-work",
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id=case_id,
        query="How many remote days?",
        relevant_chunk_ids=[RELEVANT],
        document_id=DOCUMENT_ID,
    )


def create_runner() -> tuple[RetrievalEvaluationRunner, Mock, Mock, Mock]:
    dense = Mock()
    sparse = Mock()
    hybrid = Mock()

    dense.search.return_value = DenseSearchResponse(
        query="How many remote days?",
        result_count=1,
        results=[create_result(RELEVANT)],
    )
    sparse.search.return_value = SparseSearchResponse(
        query="How many remote days?",
        result_count=1,
        results=[create_result(IRRELEVANT)],
    )
    hybrid.search.return_value = HybridSearchResponse(
        query="How many remote days?",
        result_count=1,
        results=[create_result(RELEVANT)],
    )

    return (
        RetrievalEvaluationRunner(
            dense_search_service=dense,
            sparse_search_service=sparse,
            hybrid_search_service=hybrid,
        ),
        dense,
        sparse,
        hybrid,
    )


def test_runner_calls_all_services_and_preserves_case_order() -> None:
    runner, dense, sparse, hybrid = create_runner()
    cases = [create_case("first"), create_case("second")]

    report = runner.run(cases, top_k=3, candidate_limit=10)

    assert dense.search.call_count == 2
    assert sparse.search.call_count == 2
    assert hybrid.search.call_count == 2
    assert [item.case_id for item in report.dense.evaluations] == [
        "first",
        "second",
    ]
    assert report.dense.hit_rate == 1.0
    assert report.sparse.hit_rate == 0.0
    assert report.hybrid.mean_recall == 1.0

    dense_request = dense.search.call_args_list[0].args[0]
    sparse_request = sparse.search.call_args_list[0].args[0]
    hybrid_request = hybrid.search.call_args_list[0].args[0]

    assert dense_request.limit == 3
    assert sparse_request.limit == 3
    assert hybrid_request.limit == 3
    assert hybrid_request.candidate_limit == 10
    assert dense_request.document_id == DOCUMENT_ID
    assert sparse_request.document_id == DOCUMENT_ID
    assert hybrid_request.document_id == DOCUMENT_ID


def test_runner_rejects_empty_cases() -> None:
    runner, _, _, _ = create_runner()

    with pytest.raises(ValueError, match="cases cannot be empty"):
        runner.run([])


def test_runner_rejects_duplicate_case_ids() -> None:
    runner, _, _, _ = create_runner()

    with pytest.raises(ValueError, match="case_id values must be unique"):
        runner.run([create_case("same"), create_case("same")])


def test_runner_rejects_invalid_top_k() -> None:
    runner, _, _, _ = create_runner()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        runner.run([create_case()], top_k=0)


def test_runner_rejects_candidate_limit_smaller_than_top_k() -> None:
    runner, _, _, _ = create_runner()

    with pytest.raises(ValueError, match="candidate_limit"):
        runner.run([create_case()], top_k=5, candidate_limit=4)


def test_runner_propagates_service_exceptions() -> None:
    runner, dense, _, _ = create_runner()
    dense.search.side_effect = RuntimeError("search failed")

    with pytest.raises(RuntimeError, match="search failed"):
        runner.run([create_case()])
