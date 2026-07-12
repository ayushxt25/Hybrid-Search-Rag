from pathlib import Path

import pytest

from app.ingestion.exceptions import UnsupportedFileTypeError
from app.ingestion.loaders.pdf import PDFDocumentLoader
from app.ingestion.loaders.registry import DocumentLoaderRegistry
from app.ingestion.loaders.text import TextDocumentLoader


@pytest.mark.parametrize(
    ("file_name", "expected_loader_type"),
    [
        ("policy.txt", TextDocumentLoader),
        ("guide.md", TextDocumentLoader),
        ("handbook.pdf", PDFDocumentLoader),
        ("HANDBOOK.PDF", PDFDocumentLoader),
    ],
)
def test_registry_resolves_loader_by_extension(
    file_name: str,
    expected_loader_type: type,
) -> None:
    registry = DocumentLoaderRegistry()

    loader = registry.get_loader(Path(file_name))

    assert isinstance(loader, expected_loader_type)


def test_registry_rejects_unsupported_extension() -> None:
    registry = DocumentLoaderRegistry()

    with pytest.raises(
        UnsupportedFileTypeError,
        match="Unsupported file type",
    ):
        registry.get_loader(Path("employees.csv"))
