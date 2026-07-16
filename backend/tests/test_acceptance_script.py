import json
from urllib.error import HTTPError

import pytest
from scripts.acceptance.run_acceptance import (
    AcceptanceClient,
    AcceptanceError,
    AcceptanceState,
    assert_safe_diagnostics,
    cleanup,
    parse_json,
    safe_error_detail,
)


def test_authentication_header_is_optional() -> None:
    assert "X-API-Key" not in AcceptanceClient("http://test").headers()
    assert (
        AcceptanceClient("http://test", api_key="secret").headers()["X-API-Key"]
        == "secret"
    )


def test_parse_json_requires_object() -> None:
    assert parse_json(b'{"ok": true}') == {"ok": True}
    with pytest.raises(AcceptanceError, match="JSON object"):
        parse_json(b"[1, 2]")


def test_safe_error_detail_uses_sanitized_detail() -> None:
    assert (
        safe_error_detail(json.dumps({"detail": "bad request"}).encode())
        == "bad request"
    )
    assert safe_error_detail(b"not json") == "non-json error response"


def test_diagnostic_safety_assertion() -> None:
    assert_safe_diagnostics(
        {
            "score_diagnostics": {
                "dense": {
                    "raw_score": 0.1,
                    "rank": 1,
                    "weight": 1.5,
                    "rrf_contribution": 0.02,
                },
                "sparse": {
                    "raw_score": None,
                    "rank": None,
                    "weight": 1.0,
                    "rrf_contribution": 0.0,
                },
                "fused_score": 0.02,
                "fused_rank": 1,
            }
        }
    )


def test_diagnostic_safety_rejects_vectors() -> None:
    with pytest.raises(AcceptanceError, match="vector"):
        assert_safe_diagnostics(
            {
                "score_diagnostics": {
                    "dense": {
                        "raw_score": 0.1,
                        "rank": 1,
                        "weight": 1.0,
                        "rrf_contribution": 0.0,
                    },
                    "sparse": {
                        "raw_score": None,
                        "rank": None,
                        "weight": 1.0,
                        "rrf_contribution": 0.0,
                    },
                    "fused_score": 0.1,
                    "fused_rank": 1,
                    "vector": [1.0],
                }
            }
        )


def test_cleanup_attempts_all_documents() -> None:
    calls = []

    class Client:
        def json(self, method, path, payload=None):
            calls.append((method, path))
            if path.endswith("b" * 64):
                raise AcceptanceError("already gone")

    cleanup(Client(), AcceptanceState(document_ids={"a" * 64, "b" * 64}))

    assert len(calls) == 2
    assert all(method == "DELETE" for method, _ in calls)


def test_http_error_detail_preserves_request_id() -> None:
    error = HTTPError(
        "http://test",
        500,
        "server error",
        {"X-Request-ID": "rid-1"},
        None,
    )
    assert error.headers["X-Request-ID"] == "rid-1"
