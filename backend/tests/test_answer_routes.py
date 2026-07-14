from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_grounded_answer_rate_limiter,
    get_grounded_answer_service,
)
from app.generation.models import AnswerCitation, GroundedAnswerResult
from app.generation.openai import (
    GenerationAuthenticationError,
    GenerationConnectionError,
    GenerationProviderError,
    GenerationRateLimitError,
)
from app.main import app
from app.rate_limit.models import RateLimitDecision
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

client = TestClient(app)

DOCUMENT_ID = "a" * 64
CHUNK_ID = "b" * 64


class StubGroundedAnswerService:
    def __init__(
        self,
        result: GroundedAnswerResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else create_answer_result()
        self.error = error
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error

        return self.result


class StubRateLimiter:
    def __init__(self, decision: RateLimitDecision | None = None) -> None:
        self.decision = decision or RateLimitDecision(
            allowed=True,
            limit=10,
            remaining=9,
            reset_after_seconds=60,
        )
        self.keys = []

    def check(self, key: str) -> RateLimitDecision:
        self.keys.append(key)
        return self.decision


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    app.dependency_overrides[get_grounded_answer_rate_limiter] = lambda: (
        StubRateLimiter()
    )

    yield

    app.dependency_overrides.clear()


def override_answer_service(service: StubGroundedAnswerService) -> None:
    app.dependency_overrides[get_grounded_answer_service] = lambda: service


def override_rate_limiter(limiter: StubRateLimiter) -> None:
    app.dependency_overrides[get_grounded_answer_rate_limiter] = lambda: limiter


def create_answer_result() -> GroundedAnswerResult:
    answer = "Employees may work remotely three days per week. [Source 1]"

    return GroundedAnswerResult(
        question="remote work policy",
        answer=answer,
        model_name="gpt-test",
        citations=[
            AnswerCitation(
                source_number=1,
                chunk_id=CHUNK_ID,
                document_id=DOCUMENT_ID,
                file_name="remote_policy.txt",
                heading="Remote Work",
                page_number=2,
            )
        ],
        citation_markers=[1],
        retrieved_result_count=3,
        context_source_count=1,
        context_truncated=False,
        insufficient_context=False,
        input_characters=120,
        output_characters=len(answer),
        finish_reason="completed",
    )


def create_insufficient_context_result() -> GroundedAnswerResult:
    answer = (
        "The provided documents do not contain enough information to answer this "
        "question."
    )

    return GroundedAnswerResult(
        question="unknown policy",
        answer=answer,
        model_name="not-invoked",
        citations=[],
        citation_markers=[],
        retrieved_result_count=0,
        context_source_count=0,
        context_truncated=False,
        insufficient_context=True,
        input_characters=80,
        output_characters=len(answer),
        finish_reason="insufficient_context",
    )


def post_grounded(payload: dict):
    return client.post("/api/v1/answers/grounded", json=payload)


def test_grounded_answer_returns_answer_response() -> None:
    service = StubGroundedAnswerService()
    override_answer_service(service)

    response = post_grounded(
        {
            "question": "remote work policy",
            "limit": 3,
            "candidate_limit": 10,
            "document_id": DOCUMENT_ID,
        }
    )

    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "9"
    assert response.headers["X-RateLimit-Reset"] == "60"
    body = response.json()
    assert body["answer"] == service.result.answer
    assert body["model_name"] == "gpt-test"
    assert body["retrieved_result_count"] == 3
    assert body["context_source_count"] == 1
    assert body["context_truncated"] is False
    assert body["insufficient_context"] is False
    assert body["input_characters"] == 120
    assert body["output_characters"] == len(service.result.answer)
    assert body["finish_reason"] == "completed"
    assert body["citations"] == [
        {
            "source_number": 1,
            "chunk_id": CHUNK_ID,
            "document_id": DOCUMENT_ID,
            "file_name": "remote_policy.txt",
            "heading": "Remote Work",
            "page_number": 2,
        }
    ]
    assert body["citation_markers"] == [1]


def test_grounded_answer_forwards_request_fields() -> None:
    service = StubGroundedAnswerService()
    override_answer_service(service)

    response = post_grounded(
        {
            "question": "remote work policy",
            "limit": 7,
            "candidate_limit": 25,
            "document_id": DOCUMENT_ID,
        }
    )

    assert response.status_code == 200
    request = service.requests[0]
    assert request.question == "remote work policy"
    assert request.limit == 7
    assert request.candidate_limit == 25
    assert request.document_id == DOCUMENT_ID


def test_grounded_answer_returns_insufficient_context_response() -> None:
    service = StubGroundedAnswerService(result=create_insufficient_context_result())
    override_answer_service(service)

    response = post_grounded({"question": "unknown policy"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == service.result.answer
    assert body["model_name"] == "not-invoked"
    assert body["citations"] == []
    assert body["citation_markers"] == []
    assert body["retrieved_result_count"] == 0
    assert body["context_source_count"] == 0
    assert body["insufficient_context"] is True
    assert body["finish_reason"] == "insufficient_context"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "remote work", "limit": 0},
        {"question": "remote work", "limit": 10, "candidate_limit": 5},
        {"question": "remote work", "document_id": "bad-id"},
    ],
)
def test_grounded_answer_rejects_invalid_requests_before_service(
    payload: dict,
) -> None:
    service = StubGroundedAnswerService()
    override_answer_service(service)

    response = post_grounded(payload)

    assert response.status_code == 422
    assert service.requests == []


def test_grounded_answer_maps_value_error() -> None:
    service = StubGroundedAnswerService(error=ValueError("question cannot be blank."))
    override_answer_service(service)

    response = post_grounded({"question": "remote work"})

    assert response.status_code == 422
    assert response.json() == {"detail": "question cannot be blank."}


@pytest.mark.parametrize(
    ("raised_error", "expected_status", "expected_detail"),
    [
        (
            VectorStoreConnectionError("qdrant down"),
            503,
            "The vector database is currently unavailable.",
        ),
        (
            VectorStoreConfigurationError("bad collection"),
            500,
            "Answer generation failed due to a vector-store error.",
        ),
        (
            VectorStoreDataError("bad payload"),
            500,
            "Answer generation failed due to a vector-store error.",
        ),
        (
            GenerationAuthenticationError("secret raw auth failure"),
            503,
            "The generation provider is not configured correctly.",
        ),
        (
            GenerationRateLimitError("secret raw rate failure"),
            503,
            "The generation provider is temporarily rate limited.",
        ),
        (
            GenerationConnectionError("secret raw connection failure"),
            503,
            "The generation provider is currently unavailable.",
        ),
        (
            GenerationProviderError("secret raw provider failure"),
            502,
            "The generation provider returned an error.",
        ),
        (
            RuntimeError("secret raw runtime failure"),
            500,
            "Grounded answer generation did not complete successfully.",
        ),
    ],
)
def test_grounded_answer_maps_service_errors(
    raised_error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    service = StubGroundedAnswerService(error=raised_error)
    override_answer_service(service)

    response = post_grounded({"question": "remote work"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert "secret raw" not in response.text


def test_dependency_override_prevents_real_external_access() -> None:
    service = StubGroundedAnswerService()
    override_answer_service(service)

    with (
        patch("app.api.dependencies.OpenAIGenerationProvider") as provider_class,
        patch("app.api.dependencies.QdrantVectorStore") as vector_store_class,
    ):
        response = post_grounded({"question": "remote work"})

    assert response.status_code == 200
    provider_class.assert_not_called()
    vector_store_class.assert_not_called()


def test_grounded_answer_denied_request_returns_429_and_headers() -> None:
    service = StubGroundedAnswerService()
    limiter = StubRateLimiter(
        RateLimitDecision(
            allowed=False,
            limit=10,
            remaining=0,
            reset_after_seconds=42,
        )
    )
    override_answer_service(service)
    override_rate_limiter(limiter)

    response = post_grounded({"question": "remote work"})

    assert response.status_code == 429
    assert response.json() == {
        "detail": "Too many grounded-answer requests. Please try again later."
    }
    assert response.headers["Retry-After"] == "42"
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "0"
    assert response.headers["X-RateLimit-Reset"] == "42"
    assert "X-Request-ID" in response.headers
    assert service.requests == []


def test_disabled_rate_limiting_skips_limiter() -> None:
    class FailingLimiter:
        def check(self, key: str) -> RateLimitDecision:
            raise AssertionError("limiter should not be called")

    service = StubGroundedAnswerService()
    override_answer_service(service)
    app.dependency_overrides[get_grounded_answer_rate_limiter] = FailingLimiter

    with patch(
        "app.api.routes.answers.get_settings",
        return_value=SimpleNamespace(grounded_answer_rate_limit_enabled=False),
    ):
        response = post_grounded({"question": "remote work"})

    assert response.status_code == 200
    assert "X-RateLimit-Limit" not in response.headers
    assert len(service.requests) == 1


def test_rate_limited_log_excludes_client_key(caplog) -> None:
    service = StubGroundedAnswerService()
    limiter = StubRateLimiter(
        RateLimitDecision(
            allowed=False,
            limit=10,
            remaining=0,
            reset_after_seconds=42,
        )
    )
    override_answer_service(service)
    override_rate_limiter(limiter)

    with caplog.at_level("WARNING", logger="app.grounded_answer"):
        response = post_grounded({"question": "remote work"})

    assert response.status_code == 429
    assert "testclient" not in caplog.text
    assert "remote work" not in caplog.text
    assert any(
        getattr(record, "event", None) == "grounded_answer_rate_limited"
        for record in caplog.records
    )
