import json

import pytest

from app.evaluation.loader import load_evaluation_cases

CHUNK_ID = "a" * 64


def write_dataset(
    tmp_path,
    data,
):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_dataset_loads(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "case_id": "remote-work",
                "query": "How many remote days?",
                "relevant_chunk_ids": [CHUNK_ID],
                "document_id": None,
            }
        ],
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "remote-work"


def test_missing_file_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_evaluation_cases(tmp_path / "missing.json")


def test_malformed_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_evaluation_cases(path)


def test_non_array_root_is_rejected(tmp_path) -> None:
    path = write_dataset(tmp_path, {"case_id": "remote-work"})

    with pytest.raises(ValueError, match="root must be an array"):
        load_evaluation_cases(path)


def test_empty_dataset_is_rejected(tmp_path) -> None:
    path = write_dataset(tmp_path, [])

    with pytest.raises(ValueError, match="cannot be empty"):
        load_evaluation_cases(path)


def test_invalid_case_is_rejected(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "case_id": " ",
                "query": "How many remote days?",
                "relevant_chunk_ids": [CHUNK_ID],
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid case"):
        load_evaluation_cases(path)
