import re
from collections.abc import Sequence

VALID_SOURCE_MARKER_PATTERN = re.compile(r"\[Source ([1-9][0-9]*)\]")
BRACKETED_TEXT_PATTERN = re.compile(r"\[([^\]]*)\]")


def extract_citation_markers(text: str) -> list[int]:
    """Extract exact [Source N] citation markers in occurrence order."""
    if not text.strip():
        raise ValueError("answer text cannot be blank.")

    markers: list[int] = []

    for match in BRACKETED_TEXT_PATTERN.finditer(text):
        bracketed_text = match.group(0)
        content = match.group(1)
        valid_marker = VALID_SOURCE_MARKER_PATTERN.fullmatch(bracketed_text)

        if valid_marker is not None:
            markers.append(int(valid_marker.group(1)))
            continue

        if content.lower().startswith("source"):
            raise ValueError(f"malformed Source citation marker: {bracketed_text}")

    return markers


def validate_citation_markers(
    *,
    text: str,
    available_source_numbers: Sequence[int],
    require_citations: bool,
) -> list[int]:
    """Validate generated answer citation markers against available sources."""
    source_numbers = list(available_source_numbers)

    if any(source_number <= 0 for source_number in source_numbers):
        raise ValueError("available source numbers must be positive.")

    if len(set(source_numbers)) != len(source_numbers):
        raise ValueError("available source numbers must be unique.")

    if source_numbers != sorted(source_numbers):
        raise ValueError("available source numbers must be sorted ascending.")

    markers = extract_citation_markers(text)
    available_source_number_set = set(source_numbers)

    if not source_numbers and markers:
        raise ValueError("answers without available sources cannot cite sources.")

    unknown_markers = [
        marker for marker in markers if marker not in available_source_number_set
    ]

    if unknown_markers:
        raise ValueError("citation marker references an unavailable source.")

    if require_citations and source_numbers and not markers:
        raise ValueError("at least one citation marker is required.")

    return markers
