import re


def normalize_text(text: str) -> str:
    """
    Normalize extracted document text while preserving paragraph boundaries.

    The function:
    - converts Windows and old Mac line endings to ``\\n``,
    - removes null bytes,
    - collapses repeated spaces and tabs,
    - trims whitespace around lines,
    - limits consecutive blank lines to one,
    - removes leading and trailing whitespace.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")

    cleaned_lines = []

    for line in normalized.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned_line)

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()