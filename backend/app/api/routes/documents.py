import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_document_indexing_service, require_api_key
from app.core.config import get_settings
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

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


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

            result = await run_in_threadpool(
                indexing_service.index_document,
                temporary_path,
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
