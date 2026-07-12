from pathlib import Path

import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from app.ingestion.exceptions import (
    DocumentTooLargeError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.ingestion.loaders.pdf import PDFDocumentLoader


def create_blank_pdf(path: Path, page_count: int = 1) -> None:
    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)

    with path.open("wb") as output_file:
        writer.write(output_file)


def create_text_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path))

    pdf.drawString(
        72,
        720,
        "Remote work is allowed for three days per week.",
    )
    pdf.showPage()

    pdf.drawString(
        72,
        720,
        "Employees receive eighteen paid leave days.",
    )
    pdf.showPage()

    pdf.save()


def test_pdf_loader_extracts_text_page_by_page(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "employee_handbook.pdf"
    create_text_pdf(document_path)

    loader = PDFDocumentLoader()
    document = loader.load(document_path)

    assert document.file_name == "employee_handbook.pdf"
    assert document.file_extension == ".pdf"
    assert document.source_path == document_path.resolve()
    assert document.word_count == 15

    assert len(document.sections) == 2

    first_section = document.sections[0]
    second_section = document.sections[1]

    assert first_section.section_index == 0
    assert first_section.page_number == 1
    assert first_section.heading is None
    assert first_section.content == "Remote work is allowed for three days per week."

    assert second_section.section_index == 1
    assert second_section.page_number == 2
    assert second_section.heading is None
    assert second_section.content == "Employees receive eighteen paid leave days."

    assert document.content == (
        "Remote work is allowed for three days per week.\n\n"
        "Employees receive eighteen paid leave days."
    )


def test_pdf_loader_rejects_pdf_without_extractable_text(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "scanned.pdf"
    create_blank_pdf(document_path)

    loader = PDFDocumentLoader()

    with pytest.raises(
        NoExtractableTextError,
        match="may require OCR",
    ):
        loader.load(document_path)


def test_pdf_loader_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("text", encoding="utf-8")

    loader = PDFDocumentLoader()

    with pytest.raises(
        UnsupportedFileTypeError,
        match="Unsupported file type",
    ):
        loader.load(document_path)


def test_pdf_loader_rejects_large_document(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "large.pdf"
    document_path.write_bytes(b"%PDF-1.4 test")

    loader = PDFDocumentLoader(max_file_size_bytes=5)

    with pytest.raises(
        DocumentTooLargeError,
        match="file-size limit",
    ):
        loader.load(document_path)


def test_pdf_loader_rejects_invalid_size_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="max_file_size_bytes must be greater than zero",
    ):
        PDFDocumentLoader(max_file_size_bytes=0)
