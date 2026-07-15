import logging
from time import perf_counter

from app.core.config import Settings
from app.schemas.health import ComponentHealth, ReadinessResponse
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
)
from app.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger("app.health")


class ReadinessService:
    def __init__(
        self,
        *,
        settings: Settings,
        qdrant_checker: QdrantVectorStore,
    ) -> None:
        self.settings = settings
        self.qdrant_checker = qdrant_checker

    def check(self) -> ReadinessResponse:
        started_at = perf_counter()
        if not self.settings.readiness_enabled:
            return ReadinessResponse(
                status="ready",
                components={
                    "readiness": ComponentHealth(
                        status="not_configured",
                        detail="Readiness checks are disabled.",
                    )
                },
            )

        components = {
            "qdrant": self._check_qdrant(),
            "generation": self._check_generation(),
            "embedding_configuration": self._check_embedding_configuration(),
        }
        ready = all(component.status == "healthy" for component in components.values())
        response = ReadinessResponse(
            status="ready" if ready else "not_ready",
            components=components,
        )
        self._log_completion(response, perf_counter() - started_at)
        return response

    def _check_qdrant(self) -> ComponentHealth:
        try:
            self.qdrant_checker.check_readiness()
        except VectorStoreConnectionError:
            return ComponentHealth(
                status="unhealthy",
                detail="Vector database is unavailable.",
            )
        except VectorStoreConfigurationError:
            return ComponentHealth(
                status="unhealthy",
                detail="Required hybrid collection is unavailable or incompatible.",
            )

        return ComponentHealth(status="healthy")

    def _check_generation(self) -> ComponentHealth:
        if (
            not self.settings.openai_generation_model.strip()
            or not self.settings.openai_api_key.strip()
        ):
            return ComponentHealth(
                status="unhealthy",
                detail="Generation provider is not configured.",
            )

        return ComponentHealth(status="healthy")

    def _check_embedding_configuration(self) -> ComponentHealth:
        if (
            self.qdrant_checker.vector_dimensions
            != self.settings.dense_embedding_dimensions
        ):
            return ComponentHealth(
                status="unhealthy",
                detail="Embedding dimensions are incompatible.",
            )

        return ComponentHealth(status="healthy")

    def _log_completion(
        self, response: ReadinessResponse, elapsed_seconds: float
    ) -> None:
        if not self.settings.observability_enabled:
            return

        logger.info(
            "readiness_check_completed",
            extra={
                "event": "readiness_check_completed",
                "ready": response.status == "ready",
                "qdrant_status": response.components["qdrant"].status,
                "generation_status": response.components["generation"].status,
                "embedding_configuration_status": response.components[
                    "embedding_configuration"
                ].status,
                "elapsed_ms": round(elapsed_seconds * 1000, 3),
            },
        )
