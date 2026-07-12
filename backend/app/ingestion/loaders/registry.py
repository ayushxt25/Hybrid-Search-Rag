from pathlib import Path

from app.ingestion.exceptions import UnsupportedFileTypeError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx import DOCXDocumentLoader
from app.ingestion.loaders.pdf import PDFDocumentLoader
from app.ingestion.loaders.text import TextDocumentLoader


class DocumentLoaderRegistry:
    """Resolve the correct loader for a document extension."""

    def __init__(
        self,
        loaders: list[DocumentLoader] | None = None,
    ) -> None:
        configured_loaders = loaders or [
            TextDocumentLoader(),
            PDFDocumentLoader(),
            DOCXDocumentLoader(),
        ]

        self._loaders: dict[str, DocumentLoader] = {}

        for loader in configured_loaders:
            supported_extensions = getattr(
                loader,
                "supported_extensions",
                set(),
            )

            for extension in supported_extensions:
                normalized_extension = extension.lower()

                if normalized_extension in self._loaders:
                    raise ValueError(
                        f"Duplicate loader registered for {normalized_extension}."
                    )

                self._loaders[normalized_extension] = loader

    def get_loader(self, file_path: Path) -> DocumentLoader:
        """Return the loader registered for the file extension."""
        extension = file_path.suffix.lower()
        loader = self._loaders.get(extension)

        if loader is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {extension or 'no extension'}"
            )

        return loader
