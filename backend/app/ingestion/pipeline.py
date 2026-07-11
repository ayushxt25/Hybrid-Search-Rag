from pathlib import Path

from app.ingestion.chunker import chunk_text
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.text import TextDocumentLoader
from app.schemas.document import IngestedDocument


class DocumentIngestionPipeline:
    """Coordinate document loading and chunk generation."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.loader = loader or TextDocumentLoader()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, file_path: Path) -> IngestedDocument:
        """Load a document and split every source section into chunks."""
        document = self.loader.load(file_path)

        chunks = []

        for section in document.sections:
            section_chunks = chunk_text(
                text=section.content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                section_index=section.section_index,
                page_number=section.page_number,
                heading=section.heading,
                starting_chunk_index=len(chunks),
            )
            chunks.extend(section_chunks)

        if not chunks:
            raise RuntimeError("Document loading succeeded but produced no chunks.")

        return IngestedDocument(
            document=document,
            chunks=chunks,
            chunk_count=len(chunks),
        )
