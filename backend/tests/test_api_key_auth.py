import hashlib
import hmac
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import (
    get_api_key_authenticator,
    get_grounded_answer_rate_limiter,
    get_grounded_answer_service,
)
from app.generation.models import GroundedAnswerResult
from app.main import create_application
from app.rate_limit.models import RateLimitDecision
from app.security.api_key import ApiKeyAuthenticator

VALID_KEY = "correct key"
VALID_DIGEST = hashlib.sha256(VALID_KEY.encode()).hexdigest()
ACTIVE_PATCHES = []


@pytest.fixture(autouse=True)
def stop_active_patches():
    yield

    while ACTIVE_PATCHES:
        ACTIVE_PATCHES.pop().stop()
    get_api_key_authenticator.cache_clear()


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
    def __init__(self) -> None:
        self.calls = 0

    def check(self, key: str) -> RateLimitDecision:
        self.calls += 1
        return RateLimitDecision(
            allowed=True,
            limit=10,
            remaining=9,
            reset_after_seconds=60,
        )


def make_client(**overrides):
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
        max_json_request_bytes=262144,
        observability_enabled=True,
        api_auth_enabled=True,
        api_auth_key_sha256=VALID_DIGEST,
        api_auth_header_name="X-API-Key",
        api_auth_protect_search=True,
    )
    settings.__dict__.update(overrides)
    main_settings_patch = patch("app.main.get_settings", return_value=settings)
    dependency_settings_patch = patch(
        "app.api.dependencies.get_settings",
        return_value=settings,
    )
    main_settings_patch.start()
    ACTIVE_PATCHES.append(dependency_settings_patch)
    dependency_settings_patch.start()
    get_api_key_authenticator.cache_clear()
    try:
        app = create_application()
    finally:
        main_settings_patch.stop()

    service = StubAnswerService()
    limiter = StubLimiter()
    app.dependency_overrides[get_grounded_answer_service] = lambda: service
    app.dependency_overrides[get_grounded_answer_rate_limiter] = lambda: limiter
    return TestClient(app), service, limiter


def test_valid_sha256_configuration_accepted(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_AUTH_KEY_SHA256", VALID_DIGEST.upper())

    from app.core.config import Settings

    settings = Settings(_env_file=None)

    assert settings.api_auth_key_sha256 == VALID_DIGEST


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "0" * 63, "0" * 65])
def test_invalid_digest_formats_rejected(monkeypatch, digest: str) -> None:
    monkeypatch.setenv("API_AUTH_KEY_SHA256", digest)

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_digest_rejected_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_ENABLED", "true")

    from app.core.config import Settings

    with pytest.raises(ValidationError, match="api_auth_key_sha256"):
        Settings(_env_file=None)


def test_plaintext_comparison_is_not_used() -> None:
    assert ApiKeyAuthenticator(VALID_DIGEST).authenticate(VALID_DIGEST) is False


def test_valid_wrong_missing_and_blank_keys() -> None:
    authenticator = ApiKeyAuthenticator(VALID_DIGEST)

    assert authenticator.authenticate(VALID_KEY) is True
    assert authenticator.authenticate("wrong") is False
    assert authenticator.authenticate(None) is False
    assert authenticator.authenticate(" ") is False


def test_constant_time_comparison_function_is_used() -> None:
    with patch(
        "app.security.api_key.hmac.compare_digest",
        wraps=hmac.compare_digest,
    ) as compare:
        assert ApiKeyAuthenticator(VALID_DIGEST).authenticate(VALID_KEY) is True

    compare.assert_called_once()


def test_custom_safe_header_name_works() -> None:
    client, service, _ = make_client(api_auth_header_name="X-Service-Key")

    response = client.post(
        "/api/v1/answers/grounded",
        json={"question": "safe"},
        headers={"X-Service-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert len(service.requests) == 1


def test_protected_grounded_answer_rejects_missing_key() -> None:
    client, service, limiter = make_client()

    response = client.post("/api/v1/answers/grounded", json={"question": "safe"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Valid API credentials are required."}
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert service.requests == []
    assert limiter.calls == 0


def test_valid_key_reaches_service_and_wrong_key_does_not() -> None:
    client, service, _ = make_client()

    wrong = client.post(
        "/api/v1/answers/grounded",
        json={"question": "safe"},
        headers={"X-API-Key": "wrong"},
    )
    valid = client.post(
        "/api/v1/answers/grounded",
        json={"question": "safe"},
        headers={"X-API-Key": VALID_KEY},
    )

    assert wrong.status_code == 401
    assert valid.status_code == 200
    assert len(service.requests) == 1


def test_health_endpoints_remain_public() -> None:
    client, _, _ = make_client()

    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/health/live").status_code == 200


def test_allowed_cors_origin_receives_headers_on_401() -> None:
    client, _, _ = make_client(
        cors_enabled=True,
        cors_allowed_origins=["https://app.example"],
    )

    response = client.post(
        "/api/v1/answers/grounded",
        json={"question": "safe"},
        headers={"Origin": "https://app.example"},
    )

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "https://app.example"


def test_no_key_digest_or_body_appears_in_logs(caplog) -> None:
    client, _, _ = make_client()

    with caplog.at_level("WARNING", logger="app.security"):
        response = client.post(
            "/api/v1/answers/grounded",
            json={"question": "SECRET_BODY_TEXT"},
            headers={"X-API-Key": "SECRET_KEY"},
        )

    assert response.status_code == 401
    assert "SECRET_KEY" not in caplog.text
    assert VALID_DIGEST not in caplog.text
    assert "SECRET_BODY_TEXT" not in caplog.text
    assert any(
        getattr(record, "event", None) == "authentication_rejected"
        and getattr(record, "status_code", None) == 401
        for record in caplog.records
    )
