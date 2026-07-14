from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_grounded_answer_service
from app.generation.models import GroundedAnswerResult
from app.main import app
from app.observability.middleware import REQUEST_ID_HEADER
from app.observability.request_context import get_request_id


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


@pytest.fixture(autouse=True)
def clear_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def override_service(service):
    app.dependency_overrides[get_grounded_answer_service] = lambda: service


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
