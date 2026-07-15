from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response returned by the application health endpoint."""

    status: Literal["healthy"]
    service: str
    version: str
    environment: str


class ComponentHealth(BaseModel):
    status: Literal["healthy", "unhealthy", "not_configured"]
    detail: str | None = None


class LivenessResponse(BaseModel):
    status: Literal["alive"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    components: dict[str, ComponentHealth]
