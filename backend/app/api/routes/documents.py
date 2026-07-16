import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_document_indexing_service,
    get_document_management_service,
    require_api_key,
    require_search_api_key,
)
from app.core.config import get_settings
from app.documents.service import DocumentManagementService, validate_document_id
from app.ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentDecodingError,
    DocumentLoadingError,
    DocumentTooLargeError,
    EmptyDocumentError,
    EncryptedDocumentError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.observability.request_context import get_request_id
from app.schemas.documents import (
    DocumentDeletionResponse,
    IndexedDocumentDetail,
    IndexedDocumentListResponse,
)
from app.schemas.indexing import IndexedDocumentResult
from app.services.document_indexing import DocumentIndexingService
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)
logger = logging.getLogger("app.security")
document_logger = logging.getLogger("app.documents")

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
DOCUMENT_NOT_FOUND_DETAIL = "Indexed document was not found."


def _log_document_event(
    event: str,
    **fields: object,
) -> None:
    settings = get_settings()
    if not settings.observability_enabled:
        return

    document_logger.info(
        event,
        extra={
            "event": event,
            "request_id": get_request_id(),
            **fields,
        },
    )


def _elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def _map_vector_store_error(
    error: Exception,
    *,
    detail: str,
) -> HTTPException:
    if isinstance(error, VectorStoreConnectionError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector database is currently unavailable.",
        )

    if isinstance(error, (VectorStoreConfigurationError, VectorStoreDataError)):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

    raise error


@router.get(
    "",
    response_model=IndexedDocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List indexed internal documents",
)
async def list_documents(
    _: Annotated[None, Depends(require_search_api_key)],
    management_service: Annotated[
        DocumentManagementService,
        Depends(get_document_management_service),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> IndexedDocumentListResponse:
    started_at = time.perf_counter()
    try:
        response = await run_in_threadpool(
            management_service.list_documents,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except (
        VectorStoreConnectionError,
        VectorStoreConfigurationError,
        VectorStoreDataError,
    ) as error:
        raise _map_vector_store_error(
            error,
            detail="Document listing failed due to a vector-store error.",
        ) from error

    _log_document_event(
        "document_list_completed",
        document_count=len(response.documents),
        elapsed_ms=_elapsed_ms(started_at),
    )
    return response


@router.get(
    "/{document_id}",
    response_model=IndexedDocumentDetail,
    status_code=status.HTTP_200_OK,
    summary="Get indexed internal document metadata",
)
async def get_document(
    document_id: str,
    _: Annotated[None, Depends(require_search_api_key)],
    management_service: Annotated[
        DocumentManagementService,
        Depends(get_document_management_service),
    ],
) -> IndexedDocumentDetail:
    started_at = time.perf_counter()
    try:
        detail = await run_in_threadpool(
            management_service.get_document,
            document_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except (
        VectorStoreConnectionError,
        VectorStoreConfigurationError,
        VectorStoreDataError,
    ) as error:
        raise _map_vector_store_error(
            error,
            detail="Document lookup failed due to a vector-store error.",
        ) from error

    if detail is None:
        _log_document_event(
            "document_detail_completed",
            found=False,
            chunk_count=0,
            elapsed_ms=_elapsed_ms(started_at),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOCUMENT_NOT_FOUND_DETAIL,
        )

    _log_document_event(
        "document_detail_completed",
        found=True,
        chunk_count=detail.chunk_count,
        elapsed_ms=_elapsed_ms(started_at),
    )
    return detail


@router.delete(
    "/{document_id}",
    response_model=DocumentDeletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an indexed internal document",
)
async def delete_document(
    document_id: str,
    _: Annotated[None, Depends(require_api_key)],
    management_service: Annotated[
        DocumentManagementService,
        Depends(get_document_management_service),
    ],
) -> DocumentDeletionResponse:
    started_at = time.perf_counter()
    try:
        deletion = await run_in_threadpool(
            management_service.delete_document,
            document_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except (
        VectorStoreConnectionError,
        VectorStoreConfigurationError,
        VectorStoreDataError,
    ) as error:
        raise _map_vector_store_error(
            error,
            detail="Document deletion failed due to a vector-store error.",
        ) from error

    if deletion is None:
        _log_document_event(
            "document_deleted",
            deleted=False,
            deleted_chunk_count=0,
            elapsed_ms=_elapsed_ms(started_at),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOCUMENT_NOT_FOUND_DETAIL,
        )

    _log_document_event(
        "document_deleted",
        deleted=True,
        deleted_chunk_count=deletion.deleted_chunks,
        elapsed_ms=_elapsed_ms(started_at),
    )
    return deletion


@router.post(
    "/ingest",
    response_model=IndexedDocumentResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest and index an internal document",
)
async def ingest_document(
    file: UploadFile,
    _: Annotated[None, Depends(require_api_key)],
    indexing_service: Annotated[
        DocumentIndexingService,
        Depends(get_document_indexing_service),
    ],
    management_service: Annotated[
        DocumentManagementService,
        Depends(get_document_management_service),
    ],
    replace_document_id: Annotated[str | None, Form()] = None,
) -> IndexedDocumentResult:
    """
    Validate, temporarily store, process, embed and index a document.

    Supported formats:

    - UTF-8 TXT
    - UTF-8 Markdown
    - Text-based PDF
    - Microsoft Word DOCX
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    original_file_name = Path(file.filename).name
    extension = Path(original_file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=("Only .txt, .md, .pdf and .docx files are currently supported."),
        )

    if replace_document_id is not None:
        try:
            replace_document_id = validate_document_id(replace_document_id)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

    settings = get_settings()
    contents = bytearray()
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > settings.max_document_upload_bytes:
            if settings.observability_enabled:
                logger.warning(
                    "request_rejected",
                    extra={
                        "event": "request_rejected",
                        "request_id": get_request_id(),
                        "reason": "document_too_large",
                        "status_code": status.HTTP_413_CONTENT_TOO_LARGE,
                    },
                )
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Uploaded document exceeds the configured size limit.",
            )

    await file.seek(0)

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded document is empty.",
        )

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / original_file_name
            temporary_path.write_bytes(bytes(contents))

            if replace_document_id is None:
                result = await run_in_threadpool(
                    indexing_service.index_document,
                    temporary_path,
                )
            else:
                started_at = time.perf_counter()

                def replace(path: Path) -> IndexedDocumentResult:
                    replacement, deleted_chunks = (
                        indexing_service.replace_document_for_internal_use(
                            document_path=path,
                            replace_document_id=replace_document_id,
                        )
                    )
                    _log_document_event(
                        "document_replaced",
                        deleted_chunk_count=deleted_chunks,
                        new_chunk_count=replacement.chunk_count,
                        elapsed_ms=_elapsed_ms(started_at),
                    )
                    return replacement

                result = await run_in_threadpool(
                    management_service.replace_document,
                    document_path=temporary_path,
                    replace_document_id=replace_document_id,
                    index_document=replace,
                )

    except UnsupportedFileTypeError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error

    except DocumentTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error

    except (
        EmptyDocumentError,
        DocumentDecodingError,
        EncryptedDocumentError,
        NoExtractableTextError,
        CorruptedDocumentError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    except DocumentLoadingError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
            detail="Document indexing failed due to a vector-store error.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document indexing did not complete successfully.",
        ) from error

    finally:
        await file.close()

    return result
