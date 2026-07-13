import argparse
import sys
from pathlib import Path

from app.api.dependencies import (
    get_dense_search_service,
    get_hybrid_search_service,
    get_sparse_search_service,
)
from app.evaluation.loader import load_evaluation_cases
from app.evaluation.runner import RetrievalEvaluationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense, sparse, and hybrid retrieval quality.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to a retrieval evaluation JSON dataset.",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Number of final results to evaluate.",
    )
    parser.add_argument(
        "--candidate-limit",
        default=20,
        type=int,
        help="Candidate count for hybrid dense and sparse retrieval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the full JSON evaluation report.",
    )
    return parser.parse_args()


def print_summary(report) -> None:
    print(
        f"{'Method':<8} {'HitRate@' + str(report.top_k):<11} {'MRR':<8} "
        f"{'Recall@' + str(report.top_k):<10}"
    )

    for summary in (report.dense, report.sparse, report.hybrid):
        print(
            f"{summary.method:<8} "
            f"{summary.hit_rate:<11.3f} "
            f"{summary.mean_reciprocal_rank:<8.3f} "
            f"{summary.mean_recall:<10.3f}"
        )


def main() -> int:
    args = parse_args()

    try:
        cases = load_evaluation_cases(args.dataset)
        runner = RetrievalEvaluationRunner(
            dense_search_service=get_dense_search_service(),
            sparse_search_service=get_sparse_search_service(),
            hybrid_search_service=get_hybrid_search_service(),
        )
        report = runner.run(
            cases,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
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
        print(f"Retrieval evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
