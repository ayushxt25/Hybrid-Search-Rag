from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_dense_search_service,
    get_hybrid_search_service,
    get_sparse_search_service,
)
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


@router.post(
    "/sparse",
    response_model=SparseSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search indexed document chunks lexically",
)
async def sparse_search(
    request: SparseSearchRequest,
    search_service: Annotated[
        SparseSearchService,
        Depends(get_sparse_search_service),
    ],
) -> SparseSearchResponse:
    """Encode a sparse query and return the nearest indexed chunks."""
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
    search_service: Annotated[
        HybridSearchService,
        Depends(get_hybrid_search_service),
    ],
) -> HybridSearchResponse:
    """Retrieve dense and sparse candidates, then fuse their rankings."""
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
            detail="Hybrid search failed due to a vector-store error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid search did not complete successfully.",
        ) from error
