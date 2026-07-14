from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.api import dependencies
from app.generation.models import GenerationOutput, GroundedAnswerResult
from app.main import app

QUESTION = "How many days per week may full-time employees work remotely?"
STUB_ANSWER = "Employees may work remotely up to three days per week. [Source 1]"
STUB_MODEL_NAME = "stub-grounded-provider"


class StubGenerationProvider:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationOutput:
        self.call_count += 1

        return GenerationOutput(
            text=STUB_ANSWER,
            model_name=STUB_MODEL_NAME,
            input_characters=len(system_prompt) + len(user_prompt),
            output_characters=len(STUB_ANSWER),
            finish_reason="stop",
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def corpus_document_path() -> Path:
    return (
        project_root()
        / "datasets"
        / "evaluation"
        / "corpus"
        / ("remote_work_policy.md")
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_status(response: Any, expected_status: int, action: str) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise RuntimeError(f"{action} failed with HTTP {response.status_code}.")

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"{action} returned a non-object response.")

    return data


def validate_ingestion_response(data: dict[str, Any]) -> str:
    document_id = data.get("document_id")
    require(
        isinstance(document_id, str) and len(document_id) == 64,
        "document_id must be a 64-character string.",
    )
    require(data.get("indexed_points", 0) > 0, "indexed_points must be positive.")

    return document_id


def validate_grounded_answer_response(
    data: dict[str, Any],
    *,
    document_id: str,
) -> None:
    require(data.get("question") == QUESTION, "question did not match.")
    require(data.get("answer") == STUB_ANSWER, "answer did not match stub output.")
    require(data.get("model_name") == STUB_MODEL_NAME, "model_name did not match.")
    require(data.get("finish_reason") == "stop", "finish_reason did not match.")
    require(data.get("insufficient_context") is False, "context was insufficient.")
    require(data.get("retrieved_result_count", 0) > 0, "no retrieval results.")
    require(data.get("context_source_count", 0) > 0, "no context sources.")
    require(data.get("citation_markers") == [1], "citation_markers must be [1].")
    require(
        data.get("output_characters") == len(STUB_ANSWER),
        "output_characters did not match answer length.",
    )
    require(data.get("input_characters", 0) > 0, "input_characters must be positive.")

    citations = data.get("citations")
    require(isinstance(citations, list) and citations, "citations must be non-empty.")

    first_citation = citations[0]
    require(isinstance(first_citation, dict), "first citation must be an object.")
    require(first_citation.get("source_number") == 1, "source_number must be 1.")
    require(
        first_citation.get("document_id") == document_id,
        "citation document_id did not match ingested document.",
    )
    chunk_id = first_citation.get("chunk_id")
    require(
        isinstance(chunk_id, str) and len(chunk_id) == 64,
        "citation chunk_id must be a 64-character string.",
    )


def validate_insufficient_context_response(data: dict[str, Any]) -> None:
    require(data.get("model_name") == "not-invoked", "model_name must be not-invoked.")
    require(
        data.get("finish_reason") == "insufficient_context",
        "finish_reason must be insufficient_context.",
    )
    require(data.get("citation_markers") == [], "citation_markers must be empty.")
    require(data.get("citations") == [], "citations must be empty.")


@contextmanager
def generation_provider_override(
    stub: StubGenerationProvider,
) -> Iterator[None]:
    original_factory = dependencies.get_generation_provider

    def stub_factory() -> StubGenerationProvider:
        return stub

    dependencies.get_grounded_answer_service.cache_clear()
    dependencies.get_generation_provider = stub_factory
    app.dependency_overrides[original_factory] = stub_factory

    try:
        yield
    finally:
        dependencies.get_generation_provider = original_factory
        app.dependency_overrides.pop(original_factory, None)
        dependencies.get_grounded_answer_service.cache_clear()


class InsufficientContextService:
    def answer(self, request: Any) -> GroundedAnswerResult:
        answer = (
            "The provided documents do not contain enough information to answer this "
            "question."
        )

        return GroundedAnswerResult(
            question=request.question.strip(),
            answer=answer,
            model_name="not-invoked",
            citations=[],
            citation_markers=[],
            retrieved_result_count=0,
            context_source_count=0,
            context_truncated=False,
            insufficient_context=True,
            input_characters=1,
            output_characters=len(answer),
            finish_reason="insufficient_context",
        )


def run_insufficient_context_check(
    client: TestClient,
    stub: StubGenerationProvider,
) -> None:
    before_count = stub.call_count
    app.dependency_overrides[dependencies.get_grounded_answer_service] = lambda: (
        InsufficientContextService()
    )

    try:
        response = client.post(
            "/api/v1/answers/grounded",
            json={"question": "What is the hardware reimbursement policy?"},
        )
        data = require_status(response, 200, "insufficient-context answer")
        validate_insufficient_context_response(data)
        require(stub.call_count == before_count, "stub provider was invoked.")
    finally:
        app.dependency_overrides.pop(dependencies.get_grounded_answer_service, None)


def run_smoke() -> None:
    stub = StubGenerationProvider()
    client = TestClient(app)

    with generation_provider_override(stub):
        health = client.get("/api/v1/health")
        require_status(health, 200, "health check")
        print("health passed")

        document_path = corpus_document_path()
        require(document_path.exists(), f"missing corpus document: {document_path}")
        with document_path.open("rb") as document_file:
            ingest = client.post(
                "/api/v1/documents/ingest",
                files={
                    "file": (
                        document_path.name,
                        document_file,
                        "text/markdown",
                    )
                },
            )
        document_id = validate_ingestion_response(
            require_status(ingest, 200, "document ingest")
        )
        print("document indexed")

        answer = client.post(
            "/api/v1/answers/grounded",
            json={
                "question": QUESTION,
                "limit": 3,
                "candidate_limit": 6,
                "document_id": document_id,
            },
        )
        validate_grounded_answer_response(
            require_status(answer, 200, "grounded answer"),
            document_id=document_id,
        )
        print("grounded answer passed")

        run_insufficient_context_check(client, stub)
        print("insufficient-context check passed")

    print("grounded-answer smoke test passed")


def main() -> int:
    try:
        run_smoke()
    except Exception as error:
        print(f"grounded-answer smoke test failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
