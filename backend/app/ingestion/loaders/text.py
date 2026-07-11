from pathlib import Path

from app.ingestion.exceptions import (
    DocumentDecodingError,
    DocumentNotFoundError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.normalizer import normalize_text
from app.schemas.document import LoadedDocument


class TextDocumentLoader(DocumentLoader):
    """Load UTF-8 text and Markdown documents."""

    supported_extensions = {".txt", ".md"}

    def __init__(self, max_file_size_bytes: int = 5 * 1024 * 1024) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be greater than zero.")

        self.max_file_size_bytes = max_file_size_bytes

    def load(self, file_path: Path) -> LoadedDocument:
        """Load and normalize a UTF-8 plain-text document."""
        resolved_path = file_path.resolve()

        if not resolved_path.exists():
            raise DocumentNotFoundError(f"Document does not exist: {resolved_path}")

        if not resolved_path.is_file():
            raise DocumentNotFoundError(f"Document path is not a file: {resolved_path}")

        extension = resolved_path.suffix.lower()

        if extension not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {extension or 'no extension'}"
            )

        file_size = resolved_path.stat().st_size

        if file_size > self.max_file_size_bytes:
            raise DocumentTooLargeError(
                "Document exceeds the configured file-size limit."
            )

        try:
            raw_text = resolved_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise DocumentDecodingError("Document must use UTF-8 encoding.") from error

        content = normalize_text(raw_text)

        if not content:
            raise EmptyDocumentError("Document contains no usable text.")

        return LoadedDocument(
            file_name=resolved_path.name,
            file_extension=extension,
            source_path=resolved_path,
            content=content,
            character_count=len(content),
            word_count=len(content.split()),
        )
