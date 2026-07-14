import json
from pathlib import Path
from unittest.mock import Mock

from app.evaluation.models import (
    QueryRetrievalEvaluation,
    RetrievalComparisonReport,
    RetrievalEvaluationCase,
    RetrievalMethodSummary,
)
from scripts import evaluate_retrieval

DOCUMENT_ID = "d" * 64
RELEVANT = "a" * 64


def create_report(*, document_filter_applied: bool) -> RetrievalComparisonReport:
    evaluation = QueryRetrievalEvaluation(
        case_id="remote-work",
        query="How many remote days?",
        relevant_chunk_ids=[RELEVANT],
        retrieved_chunk_ids=[RELEVANT],
        relevant_retrieved_count=1,
        hit=True,
        reciprocal_rank=1.0,
        recall=1.0,
    )
    summary = RetrievalMethodSummary(
        method="Dense",
        case_count=1,
        hit_rate=1.0,
        mean_reciprocal_rank=1.0,
        mean_recall=1.0,
        evaluations=[evaluation],
    )
    return RetrievalComparisonReport(
        top_k=1,
        document_filter_applied=document_filter_applied,
        dense=summary.model_copy(update={"method": "Dense"}),
        sparse=summary.model_copy(update={"method": "Sparse"}),
        hybrid=summary.model_copy(update={"method": "Hybrid"}),
    )


def test_parse_args_defaults_to_document_filtered_mode() -> None:
    args = evaluate_retrieval.parse_args(
        ["--dataset", "datasets/evaluation/retrieval_cases.json"]
    )

    assert args.global_search is False


def test_parse_args_global_search_selects_global_mode() -> None:
    args = evaluate_retrieval.parse_args(
        ["--dataset", "datasets/evaluation/retrieval_cases.json", "--global-search"]
    )

    assert args.global_search is True


def test_main_prints_mode_and_writes_filtered_report(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "retrieval_filtered.json"
    report = create_report(document_filter_applied=True)
    runner = Mock()
    runner.run.return_value = report

    monkeypatch.setattr(
        evaluate_retrieval,
        "load_evaluation_cases",
        Mock(
            return_value=[
                RetrievalEvaluationCase(
                    case_id="remote-work",
                    query="How many remote days?",
                    relevant_chunk_ids=[RELEVANT],
                    document_id=DOCUMENT_ID,
                )
            ]
        ),
    )
    monkeypatch.setattr(evaluate_retrieval, "get_dense_search_service", Mock())
    monkeypatch.setattr(evaluate_retrieval, "get_sparse_search_service", Mock())
    monkeypatch.setattr(evaluate_retrieval, "get_hybrid_search_service", Mock())
    monkeypatch.setattr(
        evaluate_retrieval,
        "RetrievalEvaluationRunner",
        Mock(return_value=runner),
    )

    exit_code = evaluate_retrieval.main(
        [
            "--dataset",
            "datasets/evaluation/retrieval_cases.json",
            "--top-k",
            "1",
            "--candidate-limit",
            "6",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "Evaluation mode: document-filtered" in capsys.readouterr().out
    runner.run.assert_called_once()
    assert runner.run.call_args.kwargs["use_document_filter"] is True
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["document_filter_applied"]
        is True
    )


def test_main_prints_mode_and_writes_global_report(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "retrieval_global.json"
    report = create_report(document_filter_applied=False)
    runner = Mock()
    runner.run.return_value = report

    monkeypatch.setattr(
        evaluate_retrieval,
        "load_evaluation_cases",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(evaluate_retrieval, "get_dense_search_service", Mock())
    monkeypatch.setattr(evaluate_retrieval, "get_sparse_search_service", Mock())
    monkeypatch.setattr(evaluate_retrieval, "get_hybrid_search_service", Mock())
    monkeypatch.setattr(
        evaluate_retrieval,
        "RetrievalEvaluationRunner",
        Mock(return_value=runner),
    )

    exit_code = evaluate_retrieval.main(
        [
            "--dataset",
            "datasets/evaluation/retrieval_cases.json",
            "--global-search",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "Evaluation mode: global corpus" in capsys.readouterr().out
    runner.run.assert_called_once()
    assert runner.run.call_args.kwargs["use_document_filter"] is False
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["document_filter_applied"]
        is False
    )
