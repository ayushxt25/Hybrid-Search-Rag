import json
from pathlib import Path

from pydantic import ValidationError

from app.evaluation.models import RetrievalEvaluationCase


def load_evaluation_cases(path: Path) -> list[RetrievalEvaluationCase]:
    """Load retrieval evaluation cases from a UTF-8 JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation dataset is not valid JSON: {path}") from error

    if not isinstance(raw_data, list):
        raise ValueError("Evaluation dataset root must be an array.")

    if not raw_data:
        raise ValueError("Evaluation dataset cannot be empty.")

    try:
        return [RetrievalEvaluationCase.model_validate(item) for item in raw_data]
    except ValidationError as error:
        raise ValueError("Evaluation dataset contains an invalid case.") from error
