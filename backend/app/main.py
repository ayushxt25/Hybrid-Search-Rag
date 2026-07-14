import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import shutdown_dependencies
from app.api.router import api_router
from app.core.config import get_settings
from app.observability.middleware import RequestIDMiddleware


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

    application.add_middleware(RequestIDMiddleware)

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return application


app = create_application()
