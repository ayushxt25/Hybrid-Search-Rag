import pytest

from app.context.assembler import ContextAssembler
from app.schemas.search import DenseSearchResult

DOCUMENT_ID = "d" * 64


def create_result(
    chunk_id: str,
    *,
    text: str = "Employees may work remotely.",
    file_name: str = "policy.txt",
    score: float = 0.85,
    chunk_index: int = 0,
    section_index: int = 0,
    heading: str | None = "Remote Work",
    page_number: int | None = 3,
) -> DenseSearchResult:
    return DenseSearchResult(
        point_id=f"point-{chunk_id}",
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        score=score,
        file_name=file_name,
        file_extension=".txt",
        chunk_index=chunk_index,
        section_index=section_index,
        page_number=page_number,
        heading=heading,
        text=text,
        start_word=0,
        end_word=4,
        word_count=4,
    )


def expected_block(
    *,
    source_number: int = 1,
    chunk_id: str,
    file_name: str = "policy.txt",
    heading: str = "Remote Work",
    page: str = "3",
    text: str = "Employees may work remotely.",
) -> str:
    return "\n".join(
        [
            f"[Source {source_number}]",
            f"File: {file_name}",
            f"Chunk ID: {chunk_id}",
            f"Heading: {heading}",
            f"Page: {page}",
            "Content:",
            text,
        ]
    )


def test_empty_results_return_empty_context() -> None:
    context = ContextAssembler().assemble([])

    assert context.context_text == ""
    assert context.sources == []
    assert context.source_count == 0
    assert context.total_characters == 0
    assert context.truncated is False
    assert context.omitted_result_count == 0


def test_one_result_creates_source_one_with_exact_metadata_header() -> None:
    result = create_result("a" * 64)

    context = ContextAssembler().assemble([result])

    assert context.context_text == expected_block(chunk_id=result.chunk_id)
    assert context.source_count == 1
    assert context.sources[0].source_number == 1
    assert context.sources[0].chunk_id == result.chunk_id
    assert context.sources[0].document_id == result.document_id
    assert context.sources[0].text == result.text
    assert context.total_characters == len(context.context_text)


def test_ranking_order_and_source_numbers_are_preserved() -> None:
    first = create_result("a" * 64)
    second = create_result("b" * 64)
    third = create_result("c" * 64)

    context = ContextAssembler().assemble([first, second, third])

    assert [source.chunk_id for source in context.sources] == [
        first.chunk_id,
        second.chunk_id,
        third.chunk_id,
    ]
    assert [source.source_number for source in context.sources] == [1, 2, 3]


def test_source_blocks_use_exactly_two_newlines() -> None:
    first = create_result("a" * 64)
    second = create_result("b" * 64)

    context = ContextAssembler().assemble([first, second])

    assert context.context_text == "\n\n".join(
        [
            expected_block(source_number=1, chunk_id=first.chunk_id),
            expected_block(source_number=2, chunk_id=second.chunk_id),
        ]
    )
    assert "\n\n\n" not in context.context_text


def test_headers_can_be_disabled() -> None:
    first = create_result("a" * 64, text="First chunk.")
    second = create_result("b" * 64, text="Second chunk.")

    context = ContextAssembler(include_metadata_headers=False).assemble([first, second])

    assert context.context_text == "First chunk.\n\nSecond chunk."
    assert context.sources[0].source_number == 1
    assert context.sources[1].source_number == 2


def test_max_sources_is_enforced() -> None:
    results = [
        create_result("a" * 64),
        create_result("b" * 64),
        create_result("c" * 64),
    ]

    context = ContextAssembler(max_sources=2).assemble(results)

    assert [source.chunk_id for source in context.sources] == [
        results[0].chunk_id,
        results[1].chunk_id,
    ]
    assert context.source_count == 2
    assert context.truncated is True
    assert context.omitted_result_count == 1


def test_character_budget_is_enforced_without_partial_chunks() -> None:
    first = create_result("a" * 64, text="First complete chunk.")
    second = create_result("b" * 64, text="Second complete chunk.")
    first_only_text = expected_block(
        chunk_id=first.chunk_id,
        text="First complete chunk.",
    )
    second_text = expected_block(
        source_number=2,
        chunk_id=second.chunk_id,
        text="Second complete chunk.",
    )
    budget = len(first_only_text) + 1

    context = ContextAssembler(max_characters=budget).assemble([first, second])

    assert context.context_text == first_only_text
    assert second_text not in context.context_text
    assert "Second complete chunk." not in context.context_text
    assert context.source_count == 1
    assert context.truncated is True
    assert context.omitted_result_count == 1


def test_first_oversized_source_yields_empty_context() -> None:
    results = [
        create_result("a" * 64, text="First chunk."),
        create_result("b" * 64, text="Second chunk."),
    ]

    context = ContextAssembler(max_characters=1).assemble(results)

    assert context.context_text == ""
    assert context.sources == []
    assert context.source_count == 0
    assert context.total_characters == 0
    assert context.truncated is True
    assert context.omitted_result_count == len(results)


def test_omitted_count_includes_budget_and_max_source_omissions() -> None:
    results = [
        create_result("a" * 64, text="First."),
        create_result("b" * 64, text="Second."),
        create_result("c" * 64, text="Third."),
    ]
    first_only_text = expected_block(chunk_id=results[0].chunk_id, text="First.")
    budget = len(first_only_text) + 1

    context = ContextAssembler(max_characters=budget, max_sources=2).assemble(results)

    assert context.source_count == 1
    assert context.omitted_result_count == 2
    assert context.truncated is True


def test_duplicate_chunk_ids_are_rejected() -> None:
    duplicate = "a" * 64

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        ContextAssembler().assemble(
            [
                create_result(duplicate),
                create_result(duplicate),
            ]
        )


def test_input_results_are_not_mutated() -> None:
    result = create_result("a" * 64, score=0.9)
    original = result.model_copy(deep=True)

    ContextAssembler().assemble([result])

    assert result == original


def test_missing_heading_and_page_render_as_na() -> None:
    result = create_result("a" * 64, heading=None, page_number=None)

    context = ContextAssembler().assemble([result])

    assert context.context_text == expected_block(
        chunk_id=result.chunk_id,
        heading="N/A",
        page="N/A",
    )


def test_total_characters_equals_context_text_length() -> None:
    context = ContextAssembler().assemble([create_result("a" * 64)])

    assert context.total_characters == len(context.context_text)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_characters": 0}, "max_characters must be greater than zero"),
        ({"max_characters": -1}, "max_characters must be greater than zero"),
        ({"max_sources": 0}, "max_sources must be greater than zero"),
        ({"max_sources": -1}, "max_sources must be greater than zero"),
    ],
)
def test_invalid_constructor_values_are_rejected(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ContextAssembler(**kwargs)
