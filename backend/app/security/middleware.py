import logging

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.request_context import get_request_id

logger = logging.getLogger("app.security")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, *, enabled: bool = True) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _ in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name.lower().encode("latin-1") not in existing:
                        headers.append(
                            (name.lower().encode("latin-1"), value.encode("latin-1"))
                        )
                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_with_headers)


class JsonRequestSizeLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        observability_enabled: bool,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.observability_enabled = observability_enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").split(b";", 1)[0].lower()
        if content_type != b"application/json" and not (
            content_type.startswith(b"application/") and content_type.endswith(b"+json")
        ):
            await self.app(scope, receive, send)
            return

        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                length = int(content_length.decode("ascii"))
            except ValueError:
                await self._reject(scope, receive, send, 400, "invalid_content_length")
                return
            if length < 0:
                await self._reject(scope, receive, send, 400, "invalid_content_length")
                return
            if length > self.max_bytes:
                await self._reject(scope, receive, send, 413, "json_body_too_large")
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send, 413, "json_body_too_large")
                return
            more_body = message.get("more_body", False)

        sent = False

        async def replay_receive() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        reason: str,
    ) -> None:
        if self.observability_enabled:
            logger.warning(
                "request_rejected",
                extra={
                    "event": "request_rejected",
                    "request_id": get_request_id(),
                    "reason": reason,
                    "status_code": status_code,
                },
            )
        detail = (
            "JSON request body exceeds the configured size limit."
            if status_code == 413
            else "Invalid Content-Length header."
        )
        response = JSONResponse({"detail": detail}, status_code=status_code)
        await response(scope, receive, send)
