from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response returned by the application health endpoint."""

    status: Literal["healthy"]
    service: str
    version: str
    environment: str
