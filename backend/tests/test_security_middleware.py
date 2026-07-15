from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_application


def make_settings(**overrides):
    values = {
        "app_name": "Hybrid Search RAG",
        "app_version": "0.1.0",
        "log_level": "INFO",
        "api_v1_prefix": "/api/v1",
        "trusted_hosts": ["localhost", "127.0.0.1", "testserver"],
        "cors_enabled": False,
        "cors_allowed_origins": [],
        "cors_allow_credentials": False,
        "security_headers_enabled": True,
        "max_json_request_bytes": 262144,
        "observability_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_client(**settings_overrides) -> TestClient:
    with patch(
        "app.main.get_settings", return_value=make_settings(**settings_overrides)
    ):
        return TestClient(create_application())


def test_security_headers_present_on_success() -> None:
    response = make_client().get("/api/v1/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert response.headers["Content-Security-Policy"] == (
        "default-src 'none'; frame-ancestors 'none'"
    )


def test_security_headers_present_on_handled_error() -> None:
    response = make_client().get("/missing")

    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_disabled_security_headers_are_absent() -> None:
    response = make_client(security_headers_enabled=False).get("/api/v1/health")

    assert response.status_code == 200
    assert "X-Content-Type-Options" not in response.headers


def test_valid_trusted_host_accepted() -> None:
    response = make_client().get("/api/v1/health", headers={"host": "localhost"})

    assert response.status_code == 200


def test_invalid_trusted_host_rejected_with_400() -> None:
    response = make_client().get("/api/v1/health", headers={"host": "evil.test"})

    assert response.status_code == 400
    assert "X-Request-ID" in response.headers
    assert "X-Content-Type-Options" in response.headers
    assert "evil.test" not in response.text


def test_cors_disabled_by_default() -> None:
    response = make_client().get(
        "/api/v1/health",
        headers={"origin": "https://app.example"},
    )

    assert "access-control-allow-origin" not in response.headers


def test_configured_origin_receives_cors_headers() -> None:
    response = make_client(
        cors_enabled=True,
        cors_allowed_origins=["https://app.example"],
    ).get("/api/v1/health", headers={"origin": "https://app.example"})

    assert response.headers["access-control-allow-origin"] == "https://app.example"
    assert "X-Request-ID" in response.headers["access-control-expose-headers"]


def test_unconfigured_origin_receives_no_allow_origin_header() -> None:
    response = make_client(
        cors_enabled=True,
        cors_allowed_origins=["https://app.example"],
    ).get("/api/v1/health", headers={"origin": "https://other.example"})

    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_works() -> None:
    response = make_client(
        cors_enabled=True,
        cors_allowed_origins=["https://app.example"],
    ).options(
        "/api/v1/answers/grounded",
        headers={
            "origin": "https://app.example",
            "access-control-request-method": "POST",
            "access-control-request-headers": "Content-Type, X-Request-ID",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.example"


def test_wildcard_plus_credentials_is_rejected_by_settings(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", '["*"]')
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

    with pytest.raises(ValidationError, match="wildcard"):
        Settings(_env_file=None)


def test_existing_headers_are_preserved() -> None:
    response = make_client().get(
        "/api/v1/health",
        headers={"X-Request-ID": "existing-request"},
    )

    assert response.headers["X-Request-ID"] == "existing-request"
