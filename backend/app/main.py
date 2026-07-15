import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.dependencies import shutdown_dependencies
from app.api.router import api_router
from app.core.config import get_settings
from app.observability.middleware import RequestIDMiddleware
from app.security.middleware import (
    JsonRequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level)
    logging.getLogger("app.grounded_answer").setLevel(level)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        shutdown_dependencies()


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        description=(
            "API for indexing and searching internal documents "
            "using hybrid retrieval and grounded generation."
        ),
    )

    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts,
    )
    if settings.cors_enabled:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Accept", "Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=[
                "X-Request-ID",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "Retry-After",
            ],
        )
    application.add_middleware(
        JsonRequestSizeLimitMiddleware,
        max_bytes=settings.max_json_request_bytes,
        observability_enabled=settings.observability_enabled,
    )
    application.add_middleware(
        SecurityHeadersMiddleware,
        enabled=settings.security_headers_enabled,
    )
    # Request IDs wrap handled security/size responses so rejections remain traceable.
    application.add_middleware(RequestIDMiddleware)

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return application


app = create_application()
