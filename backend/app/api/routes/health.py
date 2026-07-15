from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_readiness_service
from app.core.config import get_settings
from app.health.service import ReadinessService
from app.schemas.health import HealthResponse, LivenessResponse, ReadinessResponse

router = APIRouter(tags=["Health"])
ReadinessServiceDependency = Annotated[
    ReadinessService,
    Depends(get_readiness_service),
]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check application health",
)
def health_check() -> HealthResponse:
    """Return basic service health and version information."""
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Check application liveness",
)
def liveness_check() -> LivenessResponse:
    return LivenessResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Check application readiness",
)
def readiness_check(
    response: Response,
    service: ReadinessServiceDependency,
) -> ReadinessResponse:
    readiness = service.check()
    if readiness.status == "not_ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return readiness
