from collections.abc import Sequence

from app.schemas.search import DenseSearchResult


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[DenseSearchResult]],
    *,
    limit: int,
    k: int = 60,
) -> list[DenseSearchResult]:
    """Fuse ranked result lists using Reciprocal Rank Fusion."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    fused_scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    results_by_chunk_id: dict[str, DenseSearchResult] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()

        for rank, result in enumerate(ranked_list, start=1):
            if result.chunk_id in seen_in_list:
                continue

            seen_in_list.add(result.chunk_id)
            results_by_chunk_id.setdefault(result.chunk_id, result)
            fused_scores[result.chunk_id] = fused_scores.get(
                result.chunk_id,
                0.0,
            ) + (1.0 / (k + rank))
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

    return [
        results_by_chunk_id[chunk_id].model_copy(
            update={"score": fused_scores[chunk_id]},
        )
        for chunk_id in ranked_chunk_ids[:limit]
    ]
