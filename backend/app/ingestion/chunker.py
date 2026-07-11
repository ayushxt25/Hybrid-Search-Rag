from app.schemas.document import TextChunk


def chunk_text(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> list[TextChunk]:
    """
    Split normalized text into overlapping word-based chunks.

    Args:
        text: Non-empty normalized document text.
        chunk_size: Maximum number of words in each chunk.
        chunk_overlap: Number of words shared by adjacent chunks.

    Returns:
        Ordered chunks covering the complete input text.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = text.split()

    if not words:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start_word = 0
    chunk_index = 0

    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        chunk_words = words[start_word:end_word]
        chunk_content = " ".join(chunk_words)

        chunks.append(
            TextChunk(
                chunk_index=chunk_index,
                text=chunk_content,
                start_word=start_word,
                end_word=end_word,
                word_count=len(chunk_words),
            )
        )

        if end_word == len(words):
            break

        start_word += step
        chunk_index += 1

    return chunks
