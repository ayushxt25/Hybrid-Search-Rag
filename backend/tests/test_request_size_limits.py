from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.types import Message

from app.api.dependencies import (
    get_grounded_answer_rate_limiter,
    get_grounded_answer_service,
)
from app.generation.models import GroundedAnswerResult
from app.main import create_application
from app.rate_limit.models import RateLimitDecision
from app.security.middleware import JsonRequestSizeLimitMiddleware


class StubAnswerService:
    def __init__(self) -> None:
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        answer = "Safe answer."
        return GroundedAnswerResult(
            question=request.question,
            answer=answer,
            model_name="stub",
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


class StubLimiter:
    def check(self, key: str) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=True,
            limit=10,
            remaining=9,
            reset_after_seconds=60,
        )


def make_client(limit: int = 64):
    settings = SimpleNamespace(
        app_name="Hybrid Search RAG",
        app_version="0.1.0",
        log_level="INFO",
        api_v1_prefix="/api/v1",
        trusted_hosts=["testserver"],
        cors_enabled=False,
        cors_allowed_origins=[],
        cors_allow_credentials=False,
        security_headers_enabled=True,
        max_json_request_bytes=limit,
        observability_enabled=True,
    )
    with patch("app.main.get_settings", return_value=settings):
        app = create_application()
    service = StubAnswerService()
    app.dependency_overrides[get_grounded_answer_service] = lambda: service
    app.dependency_overrides[get_grounded_answer_rate_limiter] = StubLimiter
    return TestClient(app), service


def test_json_body_under_limit_reaches_route() -> None:
    client, service = make_client()

    response = client.post("/api/v1/answers/grounded", json={"question": "safe"})

    assert response.status_code == 200
    assert len(service.requests) == 1


def test_json_body_exactly_at_limit_is_accepted() -> None:
    body = b'{"question":"safe"}'
    client, service = make_client(limit=len(body))

    response = client.post(
        "/api/v1/answers/grounded",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert len(service.requests) == 1


def test_content_length_above_limit_returns_413_without_invoking_route() -> None:
    client, service = make_client(limit=8)

    response = client.post("/api/v1/answers/grounded", json={"question": "safe"})

    assert response.status_code == 413
    assert response.json() == {
        "detail": "JSON request body exceeds the configured size limit."
    }
    assert response.headers["X-Request-ID"]
    assert service.requests == []


async def run_middleware_request(
    *,
    headers: list[tuple[bytes, bytes]],
    messages: list[Message],
    max_bytes: int = 8,
):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = JsonRequestSizeLimitMiddleware(
        app,
        max_bytes=max_bytes,
        observability_enabled=True,
    )
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "headers": headers, "path": "/"},
        receive,
        send,
    )
    return sent


async def test_streamed_body_above_limit_returns_413(anyio_backend) -> None:
    sent = await run_middleware_request(
        headers=[(b"content-type", b"application/json")],
        messages=[
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        ],
    )

    assert sent[0]["status"] == 413


async def test_malformed_content_length_returns_400(anyio_backend) -> None:
    sent = await run_middleware_request(
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", b"abc"),
        ],
        messages=[],
    )

    assert sent[0]["status"] == 400


async def test_negative_content_length_returns_400(anyio_backend) -> None:
    sent = await run_middleware_request(
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", b"-1"),
        ],
        messages=[],
    )

    assert sent[0]["status"] == 400


def test_non_json_request_unaffected() -> None:
    client, _ = make_client(limit=1)

    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_options_unaffected() -> None:
    client, _ = make_client(limit=1)

    response = client.options("/api/v1/answers/grounded")

    assert response.status_code in {200, 405}


def test_rejection_log_contains_only_allowed_metadata(caplog) -> None:
    client, _ = make_client(limit=8)

    with caplog.at_level("WARNING", logger="app.security"):
        response = client.post(
            "/api/v1/answers/grounded",
            json={"question": "SECRET_BODY_TEXT"},
        )

    assert response.status_code == 413
    assert "SECRET_BODY_TEXT" not in caplog.text
    assert any(
        getattr(record, "event", None) == "request_rejected"
        and getattr(record, "reason", None) == "json_body_too_large"
        and getattr(record, "status_code", None) == 413
        for record in caplog.records
    )
