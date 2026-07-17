import json
import logging
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_grounded_answer_rate_limiter,
    get_grounded_answer_service,
)
from app.generation.models import GroundedAnswerResult
from app.main import app, configure_logging
from app.observability.middleware import REQUEST_ID_HEADER
from app.observability.request_context import get_request_id
from app.rate_limit.models import RateLimitDecision


class StubAnswerService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def answer(self, request):
        if self.error is not None:
            raise self.error
        answer = "Safe answer."
        return GroundedAnswerResult(
            question=request.question,
            answer=answer,
            model_name="stub-model",
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


class AllowAllLimiter:
    def check(self, key: str) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=True,
            limit=10,
            remaining=9,
            reset_after_seconds=60,
        )


@pytest.fixture(autouse=True)
def clear_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def override_service(service):
    app.dependency_overrides[get_grounded_answer_service] = lambda: service


def override_limiter(limiter):
    app.dependency_overrides[get_grounded_answer_rate_limiter] = lambda: limiter


def test_valid_incoming_request_id_is_preserved() -> None:
    override_service(StubAnswerService())
    client = TestClient(app)

    response = client.post(
        "/api/v1/answers/grounded",
        json={"question": "safe"},
        headers={REQUEST_ID_HEADER: "abc-123_X.y"},
    )

    assert response.headers[REQUEST_ID_HEADER] == "abc-123_X.y"


@pytest.mark.parametrize("header_value", [None, " ", "bad/id", "a" * 129])
def test_invalid_or_missing_request_id_is_replaced(header_value: str | None) -> None:
    override_service(StubAnswerService())
    client = TestClient(app)
    headers = {} if header_value is None else {REQUEST_ID_HEADER: header_value}

    response = client.post(
        "/api/v1/answers/grounded",
        json={"question": "safe"},
        headers=headers,
    )

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id
    assert request_id != (header_value or "")


def test_request_id_context_is_reset_after_request() -> None:
    override_service(StubAnswerService())
    client = TestClient(app)

    response = client.post("/api/v1/answers/grounded", json={"question": "safe"})

    assert response.status_code == 200
    assert get_request_id() is None


def test_response_includes_request_id_on_handled_failure() -> None:
    override_service(StubAnswerService(error=RuntimeError("secret failure")))
    client = TestClient(app)

    response = client.post("/api/v1/answers/grounded", json={"question": "safe"})

    assert response.status_code == 500
    assert response.headers[REQUEST_ID_HEADER]
    assert "secret failure" not in response.text


def test_sensitive_request_body_content_not_logged(caplog) -> None:
    service = Mock()
    service.answer.side_effect = RuntimeError("provider secret raw message")
    override_service(service)
    client = TestClient(app)

    with caplog.at_level("WARNING", logger="app.grounded_answer"):
        client.post(
            "/api/v1/answers/grounded",
            json={"question": "what is SECRET_QUESTION_TOKEN?"},
        )

    log_text = caplog.text
    assert "SECRET_QUESTION_TOKEN" not in log_text
    assert "provider secret raw message" not in log_text


def test_request_lifecycle_logs_are_structured_and_safe(caplog) -> None:
    override_service(StubAnswerService())
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="app.observability"):
        response = client.post(
            "/api/v1/answers/grounded",
            json={"question": "SECRET_QUESTION_TOKEN"},
            headers={
                REQUEST_ID_HEADER: "trace-123",
                "Authorization": "Bearer SECRET_AUTH",
                "Cookie": "session=SECRET_COOKIE",
                "X-API-Key": "SECRET_API_KEY_VALUE",
            },
        )

    assert response.status_code == 200
    events = [record for record in caplog.records if record.name == "app.observability"]
    assert [record.event for record in events] == [
        "api_request_started",
        "api_request_completed",
    ]
    completed = events[-1]
    assert completed.request_id == "trace-123"
    assert completed.method == "POST"
    assert completed.path == "/api/v1/answers/grounded"
    assert completed.status_code == 200
    assert completed.duration_ms >= 0
    assert "SECRET_API_KEY_VALUE" not in caplog.text
    assert "SECRET_AUTH" not in caplog.text
    assert "SECRET_COOKIE" not in caplog.text
    assert "SECRET_QUESTION_TOKEN" not in caplog.text


def test_configured_logging_outputs_json_without_secrets(caplog) -> None:
    configure_logging("INFO")
    override_service(StubAnswerService())
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="app.observability"):
        client.post(
            "/api/v1/answers/grounded",
            json={"question": "safe"},
            headers={"Authorization": "Bearer SECRET_AUTH"},
        )

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "api_request_completed"
    )
    formatted = logging.getLogger().handlers[0].formatter.format(record)
    payload = json.loads(formatted)

    assert payload["event"] == "api_request_completed"
    assert payload["path"] == "/api/v1/answers/grounded"
    assert "duration_ms" in payload
    assert "SECRET_AUTH" not in formatted


def test_unhandled_errors_return_safe_body_and_log_api_error(caplog) -> None:
    service = Mock()
    service.answer.side_effect = Exception("SECRET_STACK /tmp/internal/provider")
    override_service(service)
    override_limiter(AllowAllLimiter())
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="app.observability"):
        response = client.post(
            "/api/v1/answers/grounded",
            json={"question": "SECRET_QUESTION_TOKEN"},
            headers={REQUEST_ID_HEADER: "error-trace"},
        )

    assert response.status_code == 500
    assert response.headers[REQUEST_ID_HEADER] == "error-trace"
    assert response.json() == {
        "detail": "Internal server error.",
        "request_id": "error-trace",
    }
    assert "SECRET_STACK" not in response.text
    assert "/tmp/internal/provider" not in response.text
    assert "SECRET_STACK" not in caplog.text
    assert "SECRET_QUESTION_TOKEN" not in caplog.text
    assert any(
        getattr(record, "event", None) == "api_error"
        and getattr(record, "request_id", None) == "error-trace"
        and getattr(record, "duration_ms", -1) >= 0
        for record in caplog.records
    )
