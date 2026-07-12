from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_dense_search_service
from app.schemas.search_request import (
    DenseSearchRequest,
    DenseSearchResponse,
)
from app.services.dense_search import DenseSearchService
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post(
    "/dense",
    response_model=DenseSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search indexed document chunks semantically",
)
async def dense_search(
    request: DenseSearchRequest,
    search_service: Annotated[
        DenseSearchService,
        Depends(get_dense_search_service),
    ],
) -> DenseSearchResponse:
    """Embed a query and return the nearest indexed chunks."""
    try:
        return await run_in_threadpool(
            search_service.search,
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
            detail="Dense search failed due to a vector-store error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dense search did not complete successfully.",
        ) from error
