from app.ingestion.normalizer import normalize_text


def test_normalize_text_cleans_spacing_and_line_endings() -> None:
    raw_text = (
        "  Employee   Handbook\r\n"
        "\r\n"
        "\r\n"
        "Remote\twork is allowed.  \r"
        "Manager approval is required.\x00"
    )

    result = normalize_text(raw_text)

    assert result == (
        "Employee Handbook\n\n"
        "Remote work is allowed.\n"
        "Manager approval is required."
    )


def test_normalize_text_preserves_single_paragraph_break() -> None:
    raw_text = "First paragraph.\n\nSecond paragraph."

    result = normalize_text(raw_text)

    assert result == "First paragraph.\n\nSecond paragraph."


def test_normalize_text_returns_empty_string_for_whitespace() -> None:
    assert normalize_text(" \t\r\n ") == ""