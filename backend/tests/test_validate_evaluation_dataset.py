import json

import pytest

from scripts.validate_evaluation_dataset import main, validate_evaluation_dataset

DOC_A = "a" * 64
DOC_B = "b" * 64
DOC_C = "c" * 64
A_CHUNK = "1" * 64
B_CHUNK = "2" * 64
C_CHUNK = "3" * 64
UNKNOWN_DOC = "d" * 64
UNKNOWN_CHUNK = "4" * 64


def write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def create_manifest() -> dict:
    return {
        "documents": [
            {
                "file_name": "a.md",
                "document_id": DOC_A,
                "chunks": [{"chunk_id": A_CHUNK, "text": "A"}],
            },
            {
                "file_name": "b.md",
                "document_id": DOC_B,
                "chunks": [{"chunk_id": B_CHUNK, "text": "B"}],
            },
            {
                "file_name": "c.md",
                "document_id": DOC_C,
                "chunks": [{"chunk_id": C_CHUNK, "text": "C"}],
            },
        ]
    }


def create_cases() -> list[dict]:
    cases = []

    for document_index, (document_id, chunk_id) in enumerate(
        [(DOC_A, A_CHUNK), (DOC_B, B_CHUNK), (DOC_C, C_CHUNK)],
        start=1,
    ):
        for case_index in range(1, 7):
            cases.append(
                {
                    "case_id": f"doc-{document_index}-case-{case_index}",
                    "query": f"Question {document_index}-{case_index}?",
                    "relevant_chunk_ids": [chunk_id],
                    "document_id": document_id,
                }
            )

    return cases


def write_dataset_and_manifest(tmp_path, cases=None, manifest=None):
    dataset_path = tmp_path / "retrieval_cases.json"
    manifest_path = tmp_path / "manifest.json"
    write_json(dataset_path, cases if cases is not None else create_cases())
    write_json(manifest_path, manifest if manifest is not None else create_manifest())
    return dataset_path, manifest_path


def test_valid_dataset_passes(tmp_path) -> None:
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path)

    validate_evaluation_dataset(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
    )


def test_unknown_document_id_fails(tmp_path) -> None:
    cases = create_cases()
    cases[0]["document_id"] = UNKNOWN_DOC
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="Unknown document_id"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_unknown_chunk_id_fails(tmp_path) -> None:
    cases = create_cases()
    cases[0]["relevant_chunk_ids"] = [UNKNOWN_CHUNK]
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="Unknown relevant chunk_id"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_chunk_belonging_to_another_document_fails(tmp_path) -> None:
    cases = create_cases()
    cases[0]["relevant_chunk_ids"] = [B_CHUNK]
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="different document"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_duplicate_case_id_fails(tmp_path) -> None:
    cases = create_cases()
    cases[1]["case_id"] = cases[0]["case_id"]
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="Case IDs must be unique"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_duplicate_query_fails(tmp_path) -> None:
    cases = create_cases()
    cases[1]["query"] = cases[0]["query"]
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="Queries must be unique"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_duplicate_relevant_chunk_id_fails(tmp_path) -> None:
    cases = create_cases()
    cases[0]["relevant_chunk_ids"] = [A_CHUNK, A_CHUNK]
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="invalid case"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_wrong_case_count_fails(tmp_path) -> None:
    dataset_path, manifest_path = write_dataset_and_manifest(
        tmp_path, cases=create_cases()[:-1]
    )

    with pytest.raises(ValueError, match="exactly 18 cases"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_uneven_distribution_fails(tmp_path) -> None:
    cases = create_cases()
    cases[0]["document_id"] = DOC_B
    cases[0]["relevant_chunk_ids"] = [B_CHUNK]
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path, cases=cases)

    with pytest.raises(ValueError, match="exactly 6 cases"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_malformed_manifest_fails(tmp_path) -> None:
    dataset_path = tmp_path / "retrieval_cases.json"
    manifest_path = tmp_path / "manifest.json"
    write_json(dataset_path, create_cases())
    manifest_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        validate_evaluation_dataset(
            dataset_path=dataset_path, manifest_path=manifest_path
        )


def test_missing_manifest_fails(tmp_path) -> None:
    dataset_path = tmp_path / "retrieval_cases.json"
    write_json(dataset_path, create_cases())

    with pytest.raises(FileNotFoundError):
        validate_evaluation_dataset(
            dataset_path=dataset_path,
            manifest_path=tmp_path / "missing.json",
        )


def test_success_summary_is_printed_by_cli(tmp_path, capsys, monkeypatch) -> None:
    dataset_path, manifest_path = write_dataset_and_manifest(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_evaluation_dataset.py",
            "--dataset",
            str(dataset_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert main() == 0
    captured = capsys.readouterr()
    assert "Validated 18 evaluation cases across 3 documents." in captured.out


def test_failure_exits_nonzero(tmp_path, capsys, monkeypatch) -> None:
    dataset_path, manifest_path = write_dataset_and_manifest(
        tmp_path, cases=create_cases()[:-1]
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_evaluation_dataset.py",
            "--dataset",
            str(dataset_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert main() == 1
    captured = capsys.readouterr()
    assert "Evaluation dataset validation failed:" in captured.err
