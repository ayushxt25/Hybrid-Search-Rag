import pytest

from app.ingestion.chunker import chunk_text

TEST_DOCUMENT_ID = "a" * 64


def test_chunk_text_creates_overlapping_chunks() -> None:
    text = "zero one two three four five six seven eight nine"

    chunks = chunk_text(
        text=text,
        chunk_size=5,
        chunk_overlap=2,
        document_id=TEST_DOCUMENT_ID,
    )

    assert [chunk.text for chunk in chunks] == [
        "zero one two three four",
        "three four five six seven",
        "six seven eight nine",
    ]

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.start_word for chunk in chunks] == [0, 3, 6]
    assert [chunk.end_word for chunk in chunks] == [5, 8, 10]
    assert [chunk.word_count for chunk in chunks] == [5, 5, 4]
    assert [chunk.section_index for chunk in chunks] == [0, 0, 0]
    assert [chunk.page_number for chunk in chunks] == [None, None, None]

    assert all(chunk.document_id == TEST_DOCUMENT_ID for chunk in chunks)
    assert all(len(chunk.chunk_id) == 64 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_text_returns_one_chunk_for_short_text() -> None:
    chunks = chunk_text(
        text="one two three",
        chunk_size=5,
        chunk_overlap=1,
        document_id=TEST_DOCUMENT_ID,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 3
    assert chunks[0].word_count == 3
    assert chunks[0].document_id == TEST_DOCUMENT_ID
    assert len(chunks[0].chunk_id) == 64


def test_chunk_text_returns_empty_list_for_empty_text() -> None:
    assert (
        chunk_text(
            "",
            document_id=TEST_DOCUMENT_ID,
        )
        == []
    )

    assert (
        chunk_text(
            "   ",
            document_id=TEST_DOCUMENT_ID,
        )
        == []
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "expected_message"),
    [
        (0, 0, "chunk_size must be greater than zero."),
        (-1, 0, "chunk_size must be greater than zero."),
        (5, -1, "chunk_overlap cannot be negative."),
        (5, 5, "chunk_overlap must be smaller than chunk_size."),
        (5, 6, "chunk_overlap must be smaller than chunk_size."),
    ],
)
def test_chunk_text_rejects_invalid_configuration(
    chunk_size: int,
    chunk_overlap: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        chunk_text(
            text="some document text",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=TEST_DOCUMENT_ID,
        )


def test_chunk_text_preserves_source_metadata() -> None:
    chunks = chunk_text(
        text="one two three four five six",
        chunk_size=4,
        chunk_overlap=1,
        document_id=TEST_DOCUMENT_ID,
        section_index=2,
        page_number=3,
        heading="Security Policy",
        starting_chunk_index=5,
    )

    assert [chunk.chunk_index for chunk in chunks] == [5, 6]
    assert all(chunk.document_id == TEST_DOCUMENT_ID for chunk in chunks)
    assert all(chunk.section_index == 2 for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.heading == "Security Policy" for chunk in chunks)
    assert all(len(chunk.chunk_id) == 64 for chunk in chunks)


def test_chunk_text_rejects_empty_document_id() -> None:
    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        chunk_text(
            text="some document text",
            document_id="",
        )
