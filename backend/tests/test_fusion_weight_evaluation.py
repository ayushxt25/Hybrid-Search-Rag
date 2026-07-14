import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.evaluation.fusion_tuning import (
    FusionWeightConfiguration,
    FusionWeightEvaluationRunner,
)
from app.evaluation.models import RetrievalEvaluationCase
from app.schemas.search import DenseSearchResult
from app.schemas.search_request import DenseSearchResponse, SparseSearchResponse
from scripts import evaluate_fusion_weights

DOCUMENT_ID = "d" * 64
OTHER_DOCUMENT_ID = "e" * 64
RELEVANT = "z" * 64
IRRELEVANT = "a" * 64


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
    document_id: str = DOCUMENT_ID,
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id=case_id,
        query="How many remote days?",
        relevant_chunk_ids=[RELEVANT],
        document_id=document_id,
    )


def create_services() -> tuple[Mock, Mock]:
    dense = Mock()
    sparse = Mock()
    dense.search.return_value = DenseSearchResponse(
        query="How many remote days?",
        result_count=2,
        results=[create_result(RELEVANT), create_result(IRRELEVANT)],
    )
    sparse.search.return_value = SparseSearchResponse(
        query="How many remote days?",
        result_count=2,
        results=[create_result(IRRELEVANT), create_result(RELEVANT)],
    )
    return dense, sparse


def create_runner() -> tuple[FusionWeightEvaluationRunner, Mock, Mock]:
    dense, sparse = create_services()
    return (
        FusionWeightEvaluationRunner(
            dense_search_service=dense,
            sparse_search_service=sparse,
        ),
        dense,
        sparse,
    )


def create_configurations() -> list[FusionWeightConfiguration]:
    return [
        FusionWeightConfiguration(
            name="equal",
            dense_weight=1.0,
            sparse_weight=1.0,
        ),
        FusionWeightConfiguration(
            name="dense-2",
            dense_weight=2.0,
            sparse_weight=1.0,
        ),
    ]


def test_fusion_runner_queries_services_once_per_case_not_per_configuration() -> None:
    runner, dense, sparse = create_runner()
    cases = [create_case("first"), create_case("second", OTHER_DOCUMENT_ID)]

    report = runner.run(cases, configurations=create_configurations())

    assert dense.search.call_count == 2
    assert sparse.search.call_count == 2
    assert [summary.name for summary in report.configurations] == ["equal", "dense-2"]


def test_fusion_runner_dense_favored_configuration_corrects_opposing_rank_tie() -> None:
    runner, _, _ = create_runner()

    report = runner.run([create_case()], configurations=create_configurations())

    equal, dense_favored = report.configurations
    assert equal.hit_rate == 0.0
    assert equal.mean_reciprocal_rank == 0.0
    assert equal.mean_recall == 0.0
    assert dense_favored.hit_rate == 1.0
    assert dense_favored.mean_reciprocal_rank == 1.0
    assert dense_favored.mean_recall == 1.0


def test_fusion_runner_summaries_and_order_are_preserved() -> None:
    runner, _, _ = create_runner()
    configurations = create_configurations()
    cases = [create_case("first"), create_case("second", OTHER_DOCUMENT_ID)]

    report = runner.run(
        cases,
        configurations=configurations,
        top_k=1,
        candidate_limit=6,
        use_document_filter=False,
    )

    assert report.top_k == 1
    assert report.candidate_limit == 6
    assert report.document_filter_applied is False
    assert [summary.name for summary in report.configurations] == [
        configuration.name for configuration in configurations
    ]
    assert [
        evaluation.case_id for evaluation in report.configurations[0].evaluations
    ] == ["first", "second"]


def test_fusion_runner_global_mode_passes_none_document_id() -> None:
    runner, dense, sparse = create_runner()

    report = runner.run(
        [create_case()],
        configurations=create_configurations(),
        use_document_filter=False,
    )

    assert report.document_filter_applied is False
    assert dense.search.call_args.args[0].document_id is None
    assert sparse.search.call_args.args[0].document_id is None


def test_fusion_runner_filtered_mode_forwards_document_id() -> None:
    runner, dense, sparse = create_runner()

    report = runner.run(
        [create_case()],
        configurations=create_configurations(),
        use_document_filter=True,
    )

    assert report.document_filter_applied is True
    assert dense.search.call_args.args[0].document_id == DOCUMENT_ID
    assert sparse.search.call_args.args[0].document_id == DOCUMENT_ID


def test_fusion_runner_rejects_duplicate_configuration_names() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="configuration names must be unique"):
        runner.run(
            [create_case()],
            configurations=[
                FusionWeightConfiguration(
                    name="equal",
                    dense_weight=1.0,
                    sparse_weight=1.0,
                ),
                FusionWeightConfiguration(
                    name="equal",
                    dense_weight=2.0,
                    sparse_weight=1.0,
                ),
            ],
        )


def test_fusion_configuration_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="configuration name cannot be blank"):
        FusionWeightConfiguration(
            name=" ",
            dense_weight=1.0,
            sparse_weight=1.0,
        )


@pytest.mark.parametrize(
    ("dense_weight", "sparse_weight", "message"),
    [
        (0.0, 1.0, "weights must be greater than zero"),
        (-1.0, 1.0, "weights must be greater than zero"),
        (float("nan"), 1.0, "weights must be finite"),
        (float("inf"), 1.0, "weights must be finite"),
        (1.0, 0.0, "weights must be greater than zero"),
    ],
)
def test_fusion_configuration_rejects_invalid_weights(
    dense_weight: float,
    sparse_weight: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FusionWeightConfiguration(
            name="invalid",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )


def test_fusion_runner_rejects_empty_cases() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="cases cannot be empty"):
        runner.run([], configurations=create_configurations())


def test_fusion_runner_rejects_duplicate_case_ids() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="case_id values must be unique"):
        runner.run(
            [create_case("same"), create_case("same")],
            configurations=create_configurations(),
        )


def test_fusion_runner_rejects_empty_configurations() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="configurations cannot be empty"):
        runner.run([create_case()], configurations=[])


def test_fusion_runner_rejects_invalid_top_k() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        runner.run([create_case()], configurations=create_configurations(), top_k=0)


def test_fusion_runner_rejects_candidate_limit_smaller_than_top_k() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="candidate_limit"):
        runner.run(
            [create_case()],
            configurations=create_configurations(),
            top_k=5,
            candidate_limit=4,
        )


def test_fusion_runner_rejects_invalid_rrf_k() -> None:
    runner, _, _ = create_runner()

    with pytest.raises(ValueError, match="rrf_k must be greater than zero"):
        runner.run([create_case()], configurations=create_configurations(), rrf_k=0)


def test_fusion_runner_propagates_service_exceptions() -> None:
    runner, dense, _ = create_runner()
    dense.search.side_effect = RuntimeError("search failed")

    with pytest.raises(RuntimeError, match="search failed"):
        runner.run([create_case()], configurations=create_configurations())


def test_fusion_runner_inputs_are_not_mutated() -> None:
    runner, _, _ = create_runner()
    case = create_case()
    configurations = create_configurations()
    original_case = case.model_copy(deep=True)
    original_configurations = [
        configuration.model_copy(deep=True) for configuration in configurations
    ]

    runner.run([case], configurations=configurations)

    assert case == original_case
    assert configurations == original_configurations


def test_fusion_cli_parse_args_selects_global_mode() -> None:
    args = evaluate_fusion_weights.parse_args(
        ["--dataset", "datasets/evaluation/retrieval_cases.json", "--global-search"]
    )

    assert args.global_search is True


def test_fusion_cli_main_uses_runner_and_writes_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output_path = tmp_path / "fusion_weights.json"
    runner, _, _ = create_runner()
    report = runner.run([create_case()], configurations=create_configurations())
    runner_mock = Mock()
    runner_mock.run.return_value = report

    monkeypatch.setattr(
        evaluate_fusion_weights,
        "load_evaluation_cases",
        Mock(return_value=[create_case()]),
    )
    monkeypatch.setattr(evaluate_fusion_weights, "get_dense_search_service", Mock())
    monkeypatch.setattr(evaluate_fusion_weights, "get_sparse_search_service", Mock())
    monkeypatch.setattr(
        evaluate_fusion_weights,
        "FusionWeightEvaluationRunner",
        Mock(return_value=runner_mock),
    )

    exit_code = evaluate_fusion_weights.main(
        [
            "--dataset",
            "datasets/evaluation/retrieval_cases.json",
            "--top-k",
            "1",
            "--candidate-limit",
            "6",
            "--global-search",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "Configuration" in capsys.readouterr().out
    runner_mock.run.assert_called_once()
    assert runner_mock.run.call_args.kwargs["use_document_filter"] is False
    output_json = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_json["document_filter_applied"] is False
    assert output_json["configurations"][0]["name"] == "equal"
