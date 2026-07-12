from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    DocumentTooLargeError,
    EncryptedDocumentError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.ingestion.identifiers import generate_content_hash
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.normalizer import normalize_text
from app.schemas.document import DocumentSection, LoadedDocument


class PDFDocumentLoader(DocumentLoader):
    """Extract machine-readable text from PDF documents page by page."""

    supported_extensions = {".pdf"}

    def __init__(self, max_file_size_bytes: int = 20 * 1024 * 1024) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be greater than zero.")

        self.max_file_size_bytes = max_file_size_bytes

    def load(self, file_path: Path) -> LoadedDocument:
        """Load a PDF and preserve its extractable page structure."""
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
            reader = PdfReader(resolved_path)
        except PdfReadError as error:
            raise CorruptedDocumentError("PDF could not be parsed.") from error
        except OSError as error:
            raise CorruptedDocumentError("PDF could not be read.") from error

        if reader.is_encrypted:
            raise EncryptedDocumentError("Password-protected PDFs are not supported.")

        sections: list[DocumentSection] = []

        for page_index, page in enumerate(reader.pages):
            try:
                extracted_text = page.extract_text() or ""
            except (KeyError, TypeError, ValueError) as error:
                raise CorruptedDocumentError(
                    f"Unable to extract text from PDF page {page_index + 1}."
                ) from error

            section_content = normalize_text(extracted_text)

            if not section_content:
                continue

            sections.append(
                DocumentSection(
                    section_index=len(sections),
                    page_number=page_index + 1,
                    content=section_content,
                )
            )

        if not sections:
            raise NoExtractableTextError(
                "PDF contains no extractable text and may require OCR."
            )

        content = "\n\n".join(section.content for section in sections)
        content_hash = generate_content_hash(content)

        return LoadedDocument(
            document_id=content_hash,
            content_hash=content_hash,
            file_name=resolved_path.name,
            file_extension=extension,
            source_path=resolved_path,
            content=content,
            character_count=len(content),
            word_count=len(content.split()),
            sections=sections,
        )
