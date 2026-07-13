import pytest

from app.evaluation.metrics import evaluate_ranked_chunk_ids, summarize_method
from app.evaluation.models import RetrievalEvaluationCase

RELEVANT_ONE = "a" * 64
RELEVANT_TWO = "b" * 64
IRRELEVANT = "c" * 64


def create_case(
    *,
    relevant_chunk_ids: list[str] | None = None,
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id="remote-work",
        query="How many remote days?",
        relevant_chunk_ids=relevant_chunk_ids or [RELEVANT_ONE],
    )


def test_evaluates_relevant_result_at_rank_one() -> None:
    evaluation = evaluate_ranked_chunk_ids(
        case=create_case(),
        retrieved_chunk_ids=[RELEVANT_ONE, IRRELEVANT],
    )

    assert evaluation.hit is True
    assert evaluation.relevant_retrieved_count == 1
    assert evaluation.reciprocal_rank == 1.0
    assert evaluation.recall == 1.0


def test_evaluates_relevant_result_at_later_rank() -> None:
    evaluation = evaluate_ranked_chunk_ids(
        case=create_case(),
        retrieved_chunk_ids=[IRRELEVANT, RELEVANT_ONE],
    )

    assert evaluation.hit is True
    assert evaluation.reciprocal_rank == 0.5


def test_evaluates_multiple_relevant_chunks_and_recall() -> None:
    evaluation = evaluate_ranked_chunk_ids(
        case=create_case(relevant_chunk_ids=[RELEVANT_ONE, RELEVANT_TWO]),
        retrieved_chunk_ids=[RELEVANT_ONE, IRRELEVANT],
    )

    assert evaluation.relevant_retrieved_count == 1
    assert evaluation.recall == 0.5


def test_evaluates_no_relevant_results() -> None:
    evaluation = evaluate_ranked_chunk_ids(
        case=create_case(),
        retrieved_chunk_ids=[IRRELEVANT],
    )

    assert evaluation.hit is False
    assert evaluation.reciprocal_rank == 0.0
    assert evaluation.recall == 0.0


def test_evaluates_empty_retrieval() -> None:
    evaluation = evaluate_ranked_chunk_ids(
        case=create_case(),
        retrieved_chunk_ids=[],
    )

    assert evaluation.retrieved_chunk_ids == []
    assert evaluation.hit is False


def test_duplicate_retrieved_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="retrieved_chunk_ids must be unique",
    ):
        evaluate_ranked_chunk_ids(
            case=create_case(),
            retrieved_chunk_ids=[IRRELEVANT, IRRELEVANT],
        )


def test_inputs_are_not_mutated() -> None:
    case = create_case()
    retrieved = [IRRELEVANT, RELEVANT_ONE]

    evaluate_ranked_chunk_ids(
        case=case,
        retrieved_chunk_ids=retrieved,
    )

    assert case.relevant_chunk_ids == [RELEVANT_ONE]
    assert retrieved == [IRRELEVANT, RELEVANT_ONE]


def test_method_averages_are_correct() -> None:
    first = evaluate_ranked_chunk_ids(
        case=create_case(),
        retrieved_chunk_ids=[RELEVANT_ONE],
    )
    second = evaluate_ranked_chunk_ids(
        case=RetrievalEvaluationCase(
            case_id="leave",
            query="Leave days?",
            relevant_chunk_ids=[RELEVANT_TWO],
        ),
        retrieved_chunk_ids=[IRRELEVANT],
    )

    summary = summarize_method(
        method=" Dense ",
        evaluations=[first, second],
    )

    assert summary.method == "Dense"
    assert summary.case_count == 2
    assert summary.hit_rate == 0.5
    assert summary.mean_reciprocal_rank == 0.5
    assert summary.mean_recall == 0.5


def test_blank_method_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="method cannot be blank",
    ):
        summarize_method(method=" ", evaluations=[])


def test_empty_evaluations_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="evaluations cannot be empty",
    ):
        summarize_method(method="Dense", evaluations=[])
