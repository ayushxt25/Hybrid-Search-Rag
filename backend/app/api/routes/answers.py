import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_grounded_answer_rate_limiter,
    get_grounded_answer_service,
    require_api_key,
)
from app.core.config import get_settings
from app.generation.models import GroundedAnswerRequest, GroundedAnswerResult
from app.generation.openai import (
    GenerationAuthenticationError,
    GenerationConnectionError,
    GenerationProviderError,
    GenerationRateLimitError,
)
from app.generation.service import GroundedAnswerService
from app.observability.request_context import get_request_id
from app.rate_limit.in_memory import InMemoryFixedWindowRateLimiter
from app.rate_limit.models import RateLimitDecision
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

router = APIRouter(
    prefix="/answers",
    tags=["Answers"],
)
logger = logging.getLogger("app.grounded_answer")
GroundedAnswerServiceDependency = Annotated[
    GroundedAnswerService,
    Depends(get_grounded_answer_service),
]
GroundedAnswerRateLimiterDependency = Annotated[
    InMemoryFixedWindowRateLimiter,
    Depends(get_grounded_answer_rate_limiter),
]


def derive_rate_limit_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host

    return "unknown-client"


def _rate_limit_headers(decision: RateLimitDecision) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(decision.reset_after_seconds),
    }


@router.post(
    "/grounded",
    response_model=GroundedAnswerResult,
    status_code=status.HTTP_200_OK,
    summary="Generate a grounded answer from indexed documents",
)
async def grounded_answer(
    request_body: GroundedAnswerRequest,
    request: Request,
    response: Response,
    _: Annotated[None, Depends(require_api_key)],
    service: GroundedAnswerServiceDependency,
    limiter: GroundedAnswerRateLimiterDependency,
) -> GroundedAnswerResult:
    """Retrieve evidence and generate a cited grounded answer."""
    settings = get_settings()
    if settings.grounded_answer_rate_limit_enabled:
        decision = limiter.check(derive_rate_limit_key(request))
        response.headers.update(_rate_limit_headers(decision))
        if not decision.allowed:
            headers = _rate_limit_headers(decision)
            headers["Retry-After"] = str(decision.reset_after_seconds)
            if settings.observability_enabled:
                logger.warning(
                    "grounded_answer_rate_limited",
                    extra={
                        "event": "grounded_answer_rate_limited",
                        "request_id": get_request_id(),
                        "reset_after_seconds": decision.reset_after_seconds,
                    },
                )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many grounded-answer requests. Please try again later.",
                headers=headers,
            )

    try:
        return await run_in_threadpool(
            service.answer,
            request_body,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    except VectorStoreConnectionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector database is currently unavailable.",
        ) from error

    except (
        VectorStoreConfigurationError,
        VectorStoreDataError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generation failed due to a vector-store error.",
        ) from error

    except GenerationAuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The generation provider is not configured correctly.",
        ) from error

    except GenerationRateLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The generation provider is temporarily rate limited.",
        ) from error

    except GenerationConnectionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The generation provider is currently unavailable.",
        ) from error

    except GenerationProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The generation provider returned an error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grounded answer generation did not complete successfully.",
        ) from error
