from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_grounded_answer_service
from app.generation.models import GroundedAnswerRequest, GroundedAnswerResult
from app.generation.openai import (
    GenerationAuthenticationError,
    GenerationConnectionError,
    GenerationProviderError,
    GenerationRateLimitError,
)
from app.generation.service import GroundedAnswerService
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

router = APIRouter(
    prefix="/answers",
    tags=["Answers"],
)


@router.post(
    "/grounded",
    response_model=GroundedAnswerResult,
    status_code=status.HTTP_200_OK,
    summary="Generate a grounded answer from indexed documents",
)
async def grounded_answer(
    request: GroundedAnswerRequest,
    service: Annotated[
        GroundedAnswerService,
        Depends(get_grounded_answer_service),
    ],
) -> GroundedAnswerResult:
    """Retrieve evidence and generate a cited grounded answer."""
    try:
        return await run_in_threadpool(
            service.answer,
            request,
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
