import pytest

from app.ingestion.identifiers import (
    generate_chunk_id,
    generate_content_hash,
)


def test_content_hash_is_deterministic() -> None:
    first_hash = generate_content_hash("normalized content")
    second_hash = generate_content_hash("normalized content")

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_content_hash_changes_when_content_changes() -> None:
    first_hash = generate_content_hash("first content")
    second_hash = generate_content_hash("second content")

    assert first_hash != second_hash


def test_content_hash_rejects_empty_content() -> None:
    with pytest.raises(
        ValueError,
        match="Content cannot be empty",
    ):
        generate_content_hash("")


def test_chunk_id_is_deterministic() -> None:
    arguments = {
        "document_id": "a" * 64,
        "section_index": 1,
        "chunk_index": 2,
        "text": "Remote work policy.",
    }

    first_id = generate_chunk_id(**arguments)
    second_id = generate_chunk_id(**arguments)

    assert first_id == second_id
    assert len(first_id) == 64


def test_chunk_id_changes_with_chunk_position() -> None:
    first_id = generate_chunk_id(
        document_id="a" * 64,
        section_index=0,
        chunk_index=0,
        text="Remote work policy.",
    )

    second_id = generate_chunk_id(
        document_id="a" * 64,
        section_index=0,
        chunk_index=1,
        text="Remote work policy.",
    )

    assert first_id != second_id


@pytest.mark.parametrize(
    ("document_id", "section_index", "chunk_index", "text", "message"),
    [
        ("", 0, 0, "text", "document_id cannot be empty"),
        ("a" * 64, -1, 0, "text", "section_index cannot be negative"),
        ("a" * 64, 0, -1, "text", "chunk_index cannot be negative"),
        ("a" * 64, 0, 0, "", "Chunk text cannot be empty"),
    ],
)
def test_chunk_id_rejects_invalid_input(
    document_id: str,
    section_index: int,
    chunk_index: int,
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_chunk_id(
            document_id=document_id,
            section_index=section_index,
            chunk_index=chunk_index,
            text=text,
        )
