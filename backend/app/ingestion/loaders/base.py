from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import LoadedDocument


class DocumentLoader(ABC):
    """Interface implemented by all document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> LoadedDocument:
        """Extract text and metadata from a document."""
