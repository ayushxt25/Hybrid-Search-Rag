from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API for indexing and searching internal documents "
            "using hybrid retrieval and grounded generation."
        ),
    )

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return application


app = create_application()
