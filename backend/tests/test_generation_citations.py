import pytest

from app.generation.citations import (
    extract_citation_markers,
    validate_citation_markers,
)


def test_extracts_one_citation() -> None:
    assert extract_citation_markers("Use the policy. [Source 1]") == [1]


def test_extracts_multiple_citations() -> None:
    assert extract_citation_markers("A. [Source 1] B. [Source 2]") == [1, 2]


def test_preserves_duplicate_citations() -> None:
    assert extract_citation_markers("[Source 1] again [Source 1]") == [1, 1]


def test_preserves_occurrence_order() -> None:
    assert extract_citation_markers("[Source 3] then [Source 1]") == [3, 1]


def test_ignores_ordinary_non_source_brackets() -> None:
    assert extract_citation_markers("See [Appendix A] and [Note 1].") == []


def test_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="answer text cannot be blank"):
        extract_citation_markers("   ")


@pytest.mark.parametrize(
    "marker",
    [
        "[source 1]",
        "[Source]",
        "[Source 0]",
        "[Source -1]",
        "[Source 1.5]",
        "[Source abc]",
        "[Source  1]",
        "[Source 1 ]",
    ],
)
def test_rejects_malformed_source_markers(marker: str) -> None:
    with pytest.raises(ValueError, match="malformed Source citation marker"):
        extract_citation_markers(f"Answer {marker}")


def test_rejects_unknown_source_number() -> None:
    with pytest.raises(ValueError, match="unavailable source"):
        validate_citation_markers(
            text="Answer. [Source 3]",
            available_source_numbers=[1, 2],
            require_citations=True,
        )


def test_requires_citation_when_enabled_and_sources_exist() -> None:
    with pytest.raises(ValueError, match="at least one citation"):
        validate_citation_markers(
            text="Answer without markers.",
            available_source_numbers=[1],
            require_citations=True,
        )


def test_allows_no_citation_when_requirement_disabled() -> None:
    assert (
        validate_citation_markers(
            text="Answer without markers.",
            available_source_numbers=[1],
            require_citations=False,
        )
        == []
    )


def test_empty_available_sources_reject_source_markers() -> None:
    with pytest.raises(ValueError, match="without available sources"):
        validate_citation_markers(
            text="Answer. [Source 1]",
            available_source_numbers=[],
            require_citations=False,
        )


def test_validates_available_source_number_ordering() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_citation_markers(
            text="Answer. [Source 1]",
            available_source_numbers=[2, 1],
            require_citations=True,
        )


def test_validates_available_source_number_uniqueness() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_citation_markers(
            text="Answer. [Source 1]",
            available_source_numbers=[1, 1],
            require_citations=True,
        )


def test_validates_available_source_number_positivity() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_citation_markers(
            text="Answer. [Source 1]",
            available_source_numbers=[0, 1],
            require_citations=True,
        )


def test_inputs_are_not_mutated() -> None:
    source_numbers = [1, 2]
    original_source_numbers = list(source_numbers)
    text = "Answer. [Source 1]"

    markers = validate_citation_markers(
        text=text,
        available_source_numbers=source_numbers,
        require_citations=True,
    )

    assert markers == [1]
    assert source_numbers == original_source_numbers
    assert text == "Answer. [Source 1]"
