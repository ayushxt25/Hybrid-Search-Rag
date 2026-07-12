from pathlib import Path

import pytest

from app.ingestion.exceptions import (
    DocumentDecodingError,
    DocumentNotFoundError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.ingestion.loaders.text import TextDocumentLoader


def test_text_loader_loads_and_normalizes_txt_file(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "policy.txt"
    document_path.write_text(
        "  Remote   Work\r\n\r\nEmployees may work remotely.  ",
        encoding="utf-8",
    )

    loader = TextDocumentLoader()
    document = loader.load(document_path)

    assert document.file_name == "policy.txt"
    assert document.file_extension == ".txt"
    assert document.source_path == document_path.resolve()
    assert document.content == ("Remote Work\n\nEmployees may work remotely.")
    assert document.character_count == len(document.content)
    assert document.word_count == 6

    assert len(document.document_id) == 64
    assert document.content_hash == document.document_id

    assert len(document.sections) == 1
    assert document.sections[0].section_index == 0
    assert document.sections[0].page_number is None
    assert document.sections[0].heading is None
    assert document.sections[0].content == document.content


def test_text_loader_loads_markdown_file(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text(
        "# Deployment Guide\n\nRun the health check.",
        encoding="utf-8",
    )

    loader = TextDocumentLoader()
    document = loader.load(document_path)

    assert document.file_extension == ".md"
    assert document.content.startswith("# Deployment Guide")
    assert len(document.document_id) == 64
    assert document.content_hash == document.document_id
    assert len(document.sections) == 1


def test_text_loader_generates_same_id_for_same_normalized_content(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    first_path.write_text(
        "Remote   work is allowed.\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "Remote work is allowed.",
        encoding="utf-8",
    )

    loader = TextDocumentLoader()

    first_document = loader.load(first_path)
    second_document = loader.load(second_path)

    assert first_document.content == second_document.content
    assert first_document.document_id == second_document.document_id
    assert first_document.content_hash == second_document.content_hash


def test_text_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    loader = TextDocumentLoader()
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(
        DocumentNotFoundError,
        match="Document does not exist",
    ):
        loader.load(missing_path)


def test_text_loader_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "policy.csv"
    document_path.write_text("policy,data", encoding="utf-8")

    loader = TextDocumentLoader()

    with pytest.raises(
        UnsupportedFileTypeError,
        match="Unsupported file type",
    ):
        loader.load(document_path)


def test_text_loader_rejects_empty_document(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "empty.txt"
    document_path.write_text(" \n\t ", encoding="utf-8")

    loader = TextDocumentLoader()

    with pytest.raises(
        EmptyDocumentError,
        match="no usable text",
    ):
        loader.load(document_path)


def test_text_loader_rejects_large_document(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "large.txt"
    document_path.write_text("123456", encoding="utf-8")

    loader = TextDocumentLoader(max_file_size_bytes=5)

    with pytest.raises(
        DocumentTooLargeError,
        match="file-size limit",
    ):
        loader.load(document_path)


def test_text_loader_rejects_non_utf8_document(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "invalid.txt"
    document_path.write_bytes(b"\xff\xfe\xfa")

    loader = TextDocumentLoader()

    with pytest.raises(
        DocumentDecodingError,
        match="UTF-8",
    ):
        loader.load(document_path)


def test_text_loader_rejects_invalid_size_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="max_file_size_bytes must be greater than zero",
    ):
        TextDocumentLoader(max_file_size_bytes=0)
