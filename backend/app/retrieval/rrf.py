from collections.abc import Sequence
from math import isfinite

from app.schemas.search import (
    BranchScoreDiagnostic,
    DenseSearchResult,
    RetrievalScoreDiagnostic,
)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[DenseSearchResult]],
    *,
    limit: int,
    k: int = 60,
    weights: Sequence[float] | None = None,
    include_score_diagnostics: bool = False,
) -> list[DenseSearchResult]:
    """Fuse ranked result lists using Reciprocal Rank Fusion."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if weights is None:
        fusion_weights = [1.0] * len(ranked_lists)
    else:
        fusion_weights = list(weights)

        if len(fusion_weights) != len(ranked_lists):
            raise ValueError("weights must match the number of ranked lists.")

        if any(not isfinite(weight) for weight in fusion_weights):
            raise ValueError("weights must be finite.")

        if any(weight <= 0 for weight in fusion_weights):
            raise ValueError("weights must be greater than zero.")

    fused_scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    results_by_chunk_id: dict[str, DenseSearchResult] = {}
    branch_details: dict[str, dict[int, tuple[float, int, float, float]]] = {}

    for branch_index, (ranked_list, weight) in enumerate(
        zip(ranked_lists, fusion_weights, strict=True)
    ):
        seen_in_list: set[str] = set()

        for rank, result in enumerate(ranked_list, start=1):
            if result.chunk_id in seen_in_list:
                continue

            seen_in_list.add(result.chunk_id)
            results_by_chunk_id.setdefault(result.chunk_id, result)
            contribution = weight * (1.0 / (k + rank))
            fused_scores[result.chunk_id] = (
                fused_scores.get(
                    result.chunk_id,
                    0.0,
                )
                + contribution
            )
            branch_details.setdefault(result.chunk_id, {})[branch_index] = (
                result.score,
                rank,
                weight,
                contribution,
            )
            best_ranks[result.chunk_id] = min(
                best_ranks.get(result.chunk_id, rank),
                rank,
            )

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            best_ranks[chunk_id],
            chunk_id,
        ),
    )

    results = []
    for fused_rank, chunk_id in enumerate(ranked_chunk_ids[:limit], start=1):
        update = {"score": fused_scores[chunk_id]}
        if include_score_diagnostics:
            update["score_diagnostics"] = _build_score_diagnostics(
                branch_details=branch_details[chunk_id],
                fused_score=fused_scores[chunk_id],
                fused_rank=fused_rank,
                weights=fusion_weights,
            )
        results.append(results_by_chunk_id[chunk_id].model_copy(update=update))
    return results


def _build_score_diagnostics(
    *,
    branch_details: dict[int, tuple[float, int, float, float]],
    fused_score: float,
    fused_rank: int,
    weights: list[float],
) -> RetrievalScoreDiagnostic:
    return RetrievalScoreDiagnostic(
        dense=_build_branch_diagnostic(branch_details.get(0), weight=weights[0]),
        sparse=_build_branch_diagnostic(
            branch_details.get(1),
            weight=weights[1] if len(weights) > 1 else 1.0,
        ),
        fused_score=fused_score,
        fused_rank=fused_rank,
    )


def _build_branch_diagnostic(
    detail: tuple[float, int, float, float] | None,
    *,
    weight: float,
) -> BranchScoreDiagnostic:
    if detail is None:
        return BranchScoreDiagnostic(
            raw_score=None,
            rank=None,
            weight=weight,
            rrf_contribution=0.0,
        )
    raw_score, rank, weight, contribution = detail
    return BranchScoreDiagnostic(
        raw_score=raw_score,
        rank=rank,
        weight=weight,
        rrf_contribution=contribution,
    )
