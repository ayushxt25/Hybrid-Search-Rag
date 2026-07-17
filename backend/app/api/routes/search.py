import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_dense_search_service,
    get_hybrid_search_service,
    get_sparse_search_service,
    require_search_api_key,
)
from app.core.config import get_settings
from app.observability.request_context import get_request_id
from app.schemas.search_request import (
    DenseSearchRequest,
    DenseSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    SparseSearchRequest,
    SparseSearchResponse,
)
from app.services.dense_search import DenseSearchService
from app.services.hybrid_search import HybridSearchService
from app.services.sparse_search import SparseSearchService
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)
logger = logging.getLogger("app.retrieval")


def _elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def _log_retrieval_completed(
    *,
    search_type: str,
    result_count: int,
    started_at: float,
) -> None:
    settings = get_settings()
    if not settings.observability_enabled:
        return

    logger.info(
        "retrieval_completed",
        extra={
            "event": "retrieval_completed",
            "request_id": get_request_id(),
            "search_type": search_type,
            "result_count": result_count,
            "retrieval_ms": _elapsed_ms(started_at),
        },
    )


@router.post(
    "/dense",
    response_model=DenseSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search indexed document chunks semantically",
)
async def dense_search(
    request: DenseSearchRequest,
    _: Annotated[None, Depends(require_search_api_key)],
    search_service: Annotated[
        DenseSearchService,
        Depends(get_dense_search_service),
    ],
) -> DenseSearchResponse:
    """Embed a query and return the nearest indexed chunks."""
    started_at = time.perf_counter()
    try:
        response = await run_in_threadpool(
            search_service.search,
            request,
        )
        _log_retrieval_completed(
            search_type="dense",
            result_count=response.result_count,
            started_at=started_at,
        )
        return response

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


@router.post(
    "/sparse",
    response_model=SparseSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search indexed document chunks lexically",
)
async def sparse_search(
    request: SparseSearchRequest,
    _: Annotated[None, Depends(require_search_api_key)],
    search_service: Annotated[
        SparseSearchService,
        Depends(get_sparse_search_service),
    ],
) -> SparseSearchResponse:
    """Encode a sparse query and return the nearest indexed chunks."""
    started_at = time.perf_counter()
    try:
        response = await run_in_threadpool(
            search_service.search,
            request,
        )
        _log_retrieval_completed(
            search_type="sparse",
            result_count=response.result_count,
            started_at=started_at,
        )
        return response

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
            detail="Sparse search failed due to a vector-store error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sparse search did not complete successfully.",
        ) from error


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search indexed document chunks with dense and sparse fusion",
)
async def hybrid_search(
    request: HybridSearchRequest,
    _: Annotated[None, Depends(require_search_api_key)],
    search_service: Annotated[
        HybridSearchService,
        Depends(get_hybrid_search_service),
    ],
) -> HybridSearchResponse:
    """Retrieve dense and sparse candidates, then fuse their rankings."""
    started_at = time.perf_counter()
    try:
        response = await run_in_threadpool(
            search_service.search,
            request,
        )
        _log_retrieval_completed(
            search_type="hybrid",
            result_count=response.result_count,
            started_at=started_at,
        )
        return response

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
            detail="Hybrid search failed due to a vector-store error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid search did not complete successfully.",
        ) from error
