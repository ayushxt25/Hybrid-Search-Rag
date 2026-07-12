import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

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
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.schemas.document import (
    DocumentChunkResponse,
    DocumentIngestionResponse,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@router.post(
    "/ingest",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Process an internal document",
)
async def ingest_document(
    file: UploadFile,
) -> DocumentIngestionResponse:
    """
        Validate, temporarily store, normalize and chunk an uploaded document.

        The current implementation supports:
    - UTF-8 TXT files
    - UTF-8 Markdown files
    - text-based PDF files
    - Microsoft Word DOCX files

        Uploaded documents and generated chunks are not persisted yet.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=("Only .txt, .md, .pdf and .docx files are currently supported."),
        )

    contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Uploaded document exceeds the 20 MiB size limit.",
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded document is empty.",
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        pipeline = DocumentIngestionPipeline()
        result = pipeline.ingest(temporary_path)

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

    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return DocumentIngestionResponse(
        status="processed",
        document_id=result.document.document_id,
        content_hash=result.document.content_hash,
        file_name=file.filename,
        file_extension=result.document.file_extension,
        character_count=result.document.character_count,
        word_count=result.document.word_count,
        chunk_count=result.chunk_count,
        chunks=[
            DocumentChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                section_index=chunk.section_index,
                page_number=chunk.page_number,
                heading=chunk.heading,
                text=chunk.text,
                start_word=chunk.start_word,
                end_word=chunk.end_word,
                word_count=chunk.word_count,
            )
            for chunk in result.chunks
        ],
    )
