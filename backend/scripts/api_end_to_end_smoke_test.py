from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000"
QUERY = "How many days can employees work from home?"
EXPECTED_REMOTE_POLICY_PHRASES = (
    "three days per week",
    "up to three days",
    "3 days per week",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sample_document_path() -> Path:
    return project_root() / "datasets" / "sample-documents" / "remote_policy.txt"


def request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            return response.status, json.loads(response_body)
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} failed with HTTP {error.code}: {response_body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"{method} {path} failed. Is FastAPI running at {BASE_URL}?"
        ) from error


def upload_document(path: Path) -> tuple[int, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Sample document does not exist: {path}")

    boundary = f"----api-smoke-test-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "text/plain"
    file_bytes = path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )

    request = Request(
        f"{BASE_URL}/api/v1/documents/ingest",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            response_body = response.read().decode("utf-8")
            return response.status, json.loads(response_body)
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "POST /api/v1/documents/ingest failed with "
            f"HTTP {error.code}: {response_body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"POST /api/v1/documents/ingest failed. Is FastAPI running at {BASE_URL}?"
        ) from error


def assert_64_character_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field_name} must be a 64-character string: {value!r}")
    return value


def assert_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"{field_name} must be an integer greater than zero: {value!r}"
        )
    return value


def validate_ingestion_response(data: dict[str, Any]) -> tuple[str, str, int]:
    document_id = assert_64_character_string(data.get("document_id"), "document_id")
    content_hash = assert_64_character_string(data.get("content_hash"), "content_hash")
    chunk_count = assert_positive_int(data.get("chunk_count"), "chunk_count")
    indexed_points = assert_positive_int(data.get("indexed_points"), "indexed_points")

    if indexed_points != chunk_count:
        raise RuntimeError(
            "indexed_points must equal chunk_count: "
            f"indexed_points={indexed_points}, chunk_count={chunk_count}"
        )

    return document_id, content_hash, chunk_count


def validate_search_response(data: dict[str, Any]) -> None:
    result_count = assert_positive_int(data.get("result_count"), "result_count")
    results = data.get("results")

    if not isinstance(results, list) or not results:
        raise RuntimeError(f"results must contain at least one item: {results!r}")

    top_result = results[0]
    if not isinstance(top_result, dict):
        raise RuntimeError(f"top result must be an object: {top_result!r}")

    top_text = top_result.get("text")
    if not isinstance(top_text, str):
        raise RuntimeError(f"top result text must be a string: {top_text!r}")

    normalized_top_text = top_text.lower()
    if not any(
        phrase in normalized_top_text for phrase in EXPECTED_REMOTE_POLICY_PHRASES
    ):
        raise RuntimeError(
            "top result did not contain the remote-work policy rule. "
            f"Top text: {top_text!r}"
        )

    print(f"Dense search returned {result_count} result(s).")
    print(f"Top result score: {top_result.get('score')}")
    print(f"Top result text: {top_text}")


def main() -> None:
    status, health = request_json("GET", "/api/v1/health")
    print(f"GET /api/v1/health -> {status}")
    print(json.dumps(health, indent=2, sort_keys=True))

    document_path = sample_document_path()
    status, first_ingest = upload_document(document_path)
    print(f"POST /api/v1/documents/ingest -> {status}")
    print(json.dumps(first_ingest, indent=2, sort_keys=True))

    first_document_id, first_content_hash, chunk_count = validate_ingestion_response(
        first_ingest
    )
    print(f"Indexed {chunk_count} chunk(s) for document_id={first_document_id}")

    status, search = request_json(
        "POST",
        "/api/v1/search/dense",
        payload={"query": QUERY, "limit": 3},
    )
    print(f"POST /api/v1/search/dense -> {status}")
    print(json.dumps(search, indent=2, sort_keys=True))
    validate_search_response(search)

    status, second_ingest = upload_document(document_path)
    print(f"POST /api/v1/documents/ingest second upload -> {status}")
    print(json.dumps(second_ingest, indent=2, sort_keys=True))

    second_document_id, second_content_hash, _ = validate_ingestion_response(
        second_ingest
    )
    if second_document_id != first_document_id:
        raise RuntimeError(
            "document_id changed after re-upload: "
            f"first={first_document_id}, second={second_document_id}"
        )
    if second_content_hash != first_content_hash:
        raise RuntimeError(
            "content_hash changed after re-upload: "
            f"first={first_content_hash}, second={second_content_hash}"
        )

    print("API end-to-end smoke test passed.")


if __name__ == "__main__":
    main()
