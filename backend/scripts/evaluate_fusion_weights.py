import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.api.dependencies import get_dense_search_service, get_sparse_search_service
from app.evaluation.fusion_tuning import (
    FusionWeightConfiguration,
    FusionWeightEvaluationRunner,
)
from app.evaluation.loader import load_evaluation_cases

DEFAULT_CONFIGURATIONS = [
    FusionWeightConfiguration(name="equal", dense_weight=1.0, sparse_weight=1.0),
    FusionWeightConfiguration(name="dense-1.5", dense_weight=1.5, sparse_weight=1.0),
    FusionWeightConfiguration(name="dense-2", dense_weight=2.0, sparse_weight=1.0),
    FusionWeightConfiguration(name="dense-3", dense_weight=3.0, sparse_weight=1.0),
    FusionWeightConfiguration(name="sparse-1.5", dense_weight=1.0, sparse_weight=1.5),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare weighted dense/sparse RRF configurations.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to a retrieval evaluation JSON dataset.",
    )
    parser.add_argument(
        "--top-k",
        default=1,
        type=int,
        help="Number of final fused results to evaluate.",
    )
    parser.add_argument(
        "--candidate-limit",
        default=6,
        type=int,
        help="Dense and sparse candidate count to reuse for every weight.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the full JSON comparison report.",
    )
    parser.add_argument(
        "--global-search",
        action="store_true",
        help=(
            "Evaluate retrieval across the full corpus instead of filtering by "
            "document_id."
        ),
    )
    return parser.parse_args(argv)


def print_summary(report) -> None:
    print(
        f"{'Configuration':<15} {'DenseWeight':<13} {'SparseWeight':<14} "
        f"{'HitRate@' + str(report.top_k):<11} {'MRR':<8} "
        f"{'Recall@' + str(report.top_k):<10}"
    )

    for summary in report.configurations:
        print(
            f"{summary.name:<15} "
            f"{summary.dense_weight:<13.3f} "
            f"{summary.sparse_weight:<14.3f} "
            f"{summary.hit_rate:<11.3f} "
            f"{summary.mean_reciprocal_rank:<8.3f} "
            f"{summary.mean_recall:<10.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cases = load_evaluation_cases(args.dataset)
        runner = FusionWeightEvaluationRunner(
            dense_search_service=get_dense_search_service(),
            sparse_search_service=get_sparse_search_service(),
        )
        report = runner.run(
            cases,
            configurations=DEFAULT_CONFIGURATIONS,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            use_document_filter=not args.global_search,
        )

        print_summary(report)

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                report.model_dump_json(indent=2),
                encoding="utf-8",
            )

        return 0
    except Exception as error:
        print(f"Fusion weight evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
