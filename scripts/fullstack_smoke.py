from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://localhost:3000"
TIMEOUT_SECONDS = 5
RETRIES = 12


@dataclass
class SmokeResponse:
    status: int
    body: str
    headers: dict[str, str]


def fetch(path: str) -> SmokeResponse:
    request = Request(
        f"{BASE_URL}{path}", headers={"Accept": "text/html,application/json"}
    )
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return SmokeResponse(
            status=response.status,
            body=response.read().decode("utf-8", errors="replace"),
            headers={key.lower(): value for key, value in response.headers.items()},
        )


def retry_fetch(path: str) -> SmokeResponse:
    last_error: Exception | None = None
    for _ in range(RETRIES):
        try:
            return fetch(path)
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"{path} did not become available: {last_error}")


def check(name: str, fn) -> bool:
    try:
        fn()
    except Exception as error:
        print(f"[FAIL] {name}: {error}")
        return False
    print(f"[PASS] {name}")
    return True


def assert_frontend(path: str) -> None:
    response = retry_fetch(path)
    if response.status != 200 or "<html" not in response.body.lower():
        raise AssertionError(f"{path} did not return frontend HTML")


def assert_api(path: str) -> None:
    response = retry_fetch(path)
    if response.status != 200 or "{" not in response.body:
        raise AssertionError(f"{path} did not return JSON")


def assert_security_headers() -> None:
    response = retry_fetch("/")
    required = ["x-content-type-options", "referrer-policy", "content-security-policy"]
    missing = [header for header in required if header not in response.headers]
    if missing:
        raise AssertionError(f"missing headers: {', '.join(missing)}")


def main() -> int:
    checks = [
        ("Frontend root", lambda: assert_frontend("/")),
        (
            "SPA route fallback",
            lambda: (assert_frontend("/documents"), assert_frontend("/retrieval")),
        ),
        ("API proxy", lambda: assert_api("/api/v1/health/live")),
        ("Document endpoint", lambda: assert_api("/api/v1/documents")),
        ("Security headers", assert_security_headers),
    ]
    passed = sum(1 for name, fn in checks if check(name, fn))
    total = len(checks)
    if passed == total:
        print(f"Full-stack smoke passed: {passed}/{total} checks")
        return 0
    print(f"Full-stack smoke failed: {passed}/{total} checks")
    return 1


if __name__ == "__main__":
    sys.exit(main())
