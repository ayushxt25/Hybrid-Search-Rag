import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.prepare_evaluation_corpus import (
    discover_corpus_files,
    prepare_corpus_manifest,
)

DOCUMENT_ID = "d" * 64
CONTENT_HASH = "e" * 64
FIRST_CHUNK_ID = "a" * 64
SECOND_CHUNK_ID = "b" * 64


def write_file(
    directory,
    name: str,
    text: str = "content",
):
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def create_ingested_document(
    file_name: str,
    *,
    chunk_ids: list[str] | None = None,
):
    chunks = [
        SimpleNamespace(
            chunk_id=chunk_id,
            chunk_index=index,
            section_index=index,
            heading=f"Heading {index}",
            page_number=None,
            text=f"Chunk text {index}",
        )
        for index, chunk_id in enumerate(chunk_ids or [FIRST_CHUNK_ID])
    ]

    return SimpleNamespace(
        document=SimpleNamespace(
            file_name=file_name,
            document_id=DOCUMENT_ID,
            content_hash=CONTENT_HASH,
        ),
        chunk_count=len(chunks),
        chunks=chunks,
    )


def test_files_are_processed_in_sorted_order(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_file(corpus_dir, "b.md")
    write_file(corpus_dir, "a.txt")
    output = tmp_path / "manifest.json"
    service = Mock()
    service.index_document_for_internal_use.side_effect = lambda path: (
        create_ingested_document(path.name)
    )

    prepare_corpus_manifest(
        corpus_dir=corpus_dir,
        output=output,
        indexing_service=service,
    )

    assert [
        call.args[0].name
        for call in service.index_document_for_internal_use.call_args_list
    ] == ["a.txt", "b.md"]


def test_visible_unsupported_files_are_rejected_and_hidden_files_ignored(
    tmp_path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_file(corpus_dir, ".hidden.pdf")
    write_file(corpus_dir, "policy.md")
    write_file(corpus_dir, "notes.pdf")

    with pytest.raises(ValueError, match="Unsupported corpus file"):
        discover_corpus_files(corpus_dir)


def test_missing_corpus_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_corpus_files(tmp_path / "missing")


def test_empty_corpus_directory_is_rejected(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    with pytest.raises(ValueError, match="contains no .md or .txt files"):
        discover_corpus_files(corpus_dir)


def test_existing_output_without_force_is_rejected(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_file(corpus_dir, "policy.md")
    output = tmp_path / "manifest.json"
    output.write_text("{}", encoding="utf-8")
    service = Mock()

    with pytest.raises(FileExistsError, match="Use --force"):
        prepare_corpus_manifest(
            corpus_dir=corpus_dir,
            output=output,
            indexing_service=service,
        )

    service.index_document_for_internal_use.assert_not_called()


def test_force_allows_replacement(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_file(corpus_dir, "policy.md")
    output = tmp_path / "manifest.json"
    output.write_text("old", encoding="utf-8")
    service = Mock()
    service.index_document_for_internal_use.return_value = create_ingested_document(
        "policy.md"
    )

    prepare_corpus_manifest(
        corpus_dir=corpus_dir,
        output=output,
        force=True,
        indexing_service=service,
    )

    assert json.loads(output.read_text(encoding="utf-8"))["document_count"] == 1


def test_manifest_contains_real_values_and_preserves_chunk_metadata(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_file(corpus_dir, "policy.md")
    output = tmp_path / "nested" / "manifest.json"
    service = Mock()
    service.index_document_for_internal_use.return_value = create_ingested_document(
        "policy.md",
        chunk_ids=[FIRST_CHUNK_ID, SECOND_CHUNK_ID],
    )

    manifest = prepare_corpus_manifest(
        corpus_dir=corpus_dir,
        output=output,
        indexing_service=service,
    )

    assert output.exists()
    assert manifest["corpus_directory"] == corpus_dir.as_posix()
    assert manifest["document_count"] == 1
    assert manifest["chunk_count"] == 2
    document = manifest["documents"][0]
    assert document["file_name"] == "policy.md"
    assert document["document_id"] == DOCUMENT_ID
    assert document["content_hash"] == CONTENT_HASH
    assert document["chunk_count"] == 2
    assert document["chunks"][0] == {
        "chunk_id": FIRST_CHUNK_ID,
        "chunk_index": 0,
        "section_index": 0,
        "heading": "Heading 0",
        "page_number": None,
        "text": "Chunk text 0",
    }
