import logging
import re
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.observability.logging import utc_timestamp
from app.observability.request_context import reset_request_id, set_request_id

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
logger = logging.getLogger("app.observability")


def normalize_request_id(value: str | None) -> str:
    if value is not None and _SAFE_REQUEST_ID.fullmatch(value.strip()):
        return value.strip()
    return str(uuid4())


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, observability_enabled: bool = True) -> None:
        super().__init__(app)
        self.observability_enabled = observability_enabled

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        token = set_request_id(request_id)
        started_at = time.perf_counter()
        method = request.method
        path = request.url.path
        try:
            if self.observability_enabled:
                self._log_request_started(
                    request_id=request_id,
                    method=method,
                    path=path,
                )
            try:
                response = await call_next(request)
            except Exception:
                duration_ms = self._duration_ms(started_at)
                if self.observability_enabled:
                    self._log_api_error(
                        request_id=request_id,
                        method=method,
                        path=path,
                        duration_ms=duration_ms,
                    )
                response = JSONResponse(
                    {
                        "detail": "Internal server error.",
                        "request_id": request_id,
                    },
                    status_code=500,
                )
            response.headers[REQUEST_ID_HEADER] = request_id
            if self.observability_enabled:
                self._log_request_completed(
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    duration_ms=self._duration_ms(started_at),
                )
            return response
        finally:
            reset_request_id(token)

    @staticmethod
    def _duration_ms(started_at: float) -> int:
        return round((time.perf_counter() - started_at) * 1000)

    @staticmethod
    def _log_request_started(
        *,
        request_id: str,
        method: str,
        path: str,
    ) -> None:
        logger.info(
            "api_request_started",
            extra={
                "event": "api_request_started",
                "request_id": request_id,
                "method": method,
                "path": path,
                "timestamp": utc_timestamp(),
            },
        )

    @staticmethod
    def _log_request_completed(
        *,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: int,
    ) -> None:
        logger.info(
            "api_request_completed",
            extra={
                "event": "api_request_completed",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "timestamp": utc_timestamp(),
            },
        )

    @staticmethod
    def _log_api_error(
        *,
        request_id: str,
        method: str,
        path: str,
        duration_ms: int,
    ) -> None:
        logger.error(
            "api_error",
            extra={
                "event": "api_error",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": 500,
                "duration_ms": duration_ms,
                "timestamp": utc_timestamp(),
            },
        )
