import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.loader import load_evaluation_cases  # noqa: E402

EXPECTED_CASE_COUNT = 18
EXPECTED_CASES_PER_DOCUMENT = 6


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest is not valid JSON: {path}") from error

    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be an object.")

    documents = manifest.get("documents")

    if not isinstance(documents, list) or not documents:
        raise ValueError("Manifest must contain a non-empty documents array.")

    return manifest


def build_manifest_indexes(
    manifest: dict[str, Any],
) -> tuple[set[str], dict[str, str]]:
    document_ids: set[str] = set()
    chunk_to_document_id: dict[str, str] = {}

    for document in manifest["documents"]:
        document_id = document.get("document_id")
        chunks = document.get("chunks")

        if not isinstance(document_id, str) or len(document_id) != 64:
            raise ValueError("Manifest contains an invalid document_id.")

        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"Manifest document has no chunks: {document_id}")

        document_ids.add(document_id)

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or len(chunk_id) != 64:
                raise ValueError("Manifest contains an invalid chunk_id.")

            if chunk_id in chunk_to_document_id:
                raise ValueError(f"Manifest contains duplicate chunk_id: {chunk_id}")

            chunk_to_document_id[chunk_id] = document_id

    return document_ids, chunk_to_document_id


def validate_evaluation_dataset(
    *,
    dataset_path: Path,
    manifest_path: Path,
) -> None:
    cases = load_evaluation_cases(dataset_path)
    manifest = load_manifest(manifest_path)
    document_ids, chunk_to_document_id = build_manifest_indexes(manifest)

    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Dataset must contain exactly {EXPECTED_CASE_COUNT} cases.")

    case_ids = [case.case_id for case in cases]
    queries = [case.query for case in cases]

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique.")

    if len(set(queries)) != len(queries):
        raise ValueError("Queries must be unique.")

    cases_by_document_id: Counter[str] = Counter()

    for case in cases:
        if not case.query.strip():
            raise ValueError(f"Query cannot be blank: {case.case_id}")

        if case.document_id not in document_ids:
            raise ValueError(f"Unknown document_id in case: {case.case_id}")

        if len(set(case.relevant_chunk_ids)) != len(case.relevant_chunk_ids):
            raise ValueError(f"Duplicate relevant chunk ID in case: {case.case_id}")

        cases_by_document_id[case.document_id] += 1

        for chunk_id in case.relevant_chunk_ids:
            owner_document_id = chunk_to_document_id.get(chunk_id)

            if owner_document_id is None:
                raise ValueError(f"Unknown relevant chunk_id in case: {case.case_id}")

            if owner_document_id != case.document_id:
                raise ValueError(
                    "Relevant chunk belongs to a different document in case: "
                    f"{case.case_id}"
                )

    if set(cases_by_document_id) != document_ids:
        raise ValueError("Every manifest document must have evaluation cases.")

    uneven_document_ids = [
        document_id
        for document_id, count in cases_by_document_id.items()
        if count != EXPECTED_CASES_PER_DOCUMENT
    ]

    if uneven_document_ids:
        raise ValueError(
            f"Each document must have exactly {EXPECTED_CASES_PER_DOCUMENT} cases."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the golden retrieval evaluation dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to retrieval_cases.json.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to evaluation_corpus_manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        validate_evaluation_dataset(
            dataset_path=args.dataset,
            manifest_path=args.manifest,
        )
    except Exception as error:
        print(f"Evaluation dataset validation failed: {error}", file=sys.stderr)
        return 1

    print("Validated 18 evaluation cases across 3 documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
