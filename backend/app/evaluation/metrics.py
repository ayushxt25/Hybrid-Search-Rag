from collections.abc import Sequence

from app.evaluation.models import (
    QueryRetrievalEvaluation,
    RetrievalEvaluationCase,
    RetrievalMethodSummary,
)


def evaluate_ranked_chunk_ids(
    *,
    case: RetrievalEvaluationCase,
    retrieved_chunk_ids: Sequence[str],
) -> QueryRetrievalEvaluation:
    """Evaluate one ranked chunk-id list against one golden case."""
    retrieved = list(retrieved_chunk_ids)

    if len(set(retrieved)) != len(retrieved):
        raise ValueError("retrieved_chunk_ids must be unique.")

    relevant = set(case.relevant_chunk_ids)
    relevant_retrieved_count = sum(1 for chunk_id in retrieved if chunk_id in relevant)
    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in relevant
        ),
        None,
    )

    return QueryRetrievalEvaluation(
        case_id=case.case_id,
        query=case.query,
        relevant_chunk_ids=list(case.relevant_chunk_ids),
        retrieved_chunk_ids=retrieved,
        relevant_retrieved_count=relevant_retrieved_count,
        hit=first_relevant_rank is not None,
        reciprocal_rank=(
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        ),
        recall=relevant_retrieved_count / len(case.relevant_chunk_ids),
    )


def summarize_method(
    *,
    method: str,
    evaluations: Sequence[QueryRetrievalEvaluation],
) -> RetrievalMethodSummary:
    """Summarize per-query retrieval evaluations for one method."""
    normalized_method = method.strip()

    if not normalized_method:
        raise ValueError("method cannot be blank.")

    if not evaluations:
        raise ValueError("evaluations cannot be empty.")

    evaluation_list = list(evaluations)
    case_count = len(evaluation_list)

    return RetrievalMethodSummary(
        method=normalized_method,
        case_count=case_count,
        hit_rate=sum(1 for evaluation in evaluation_list if evaluation.hit)
        / case_count,
        mean_reciprocal_rank=sum(
            evaluation.reciprocal_rank for evaluation in evaluation_list
        )
        / case_count,
        mean_recall=sum(evaluation.recall for evaluation in evaluation_list)
        / case_count,
        evaluations=evaluation_list,
    )
