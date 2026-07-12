from app.ingestion.identifiers import generate_chunk_id
from app.schemas.document import TextChunk


def chunk_text(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    *,
    document_id: str,
    section_index: int = 0,
    page_number: int | None = None,
    heading: str | None = None,
    starting_chunk_index: int = 0,
) -> list[TextChunk]:
    """
    Split text into overlapping word-based chunks.

    Word offsets are relative to the source section.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if not document_id:
        raise ValueError("document_id cannot be empty.")

    if section_index < 0:
        raise ValueError("section_index cannot be negative.")

    if starting_chunk_index < 0:
        raise ValueError("starting_chunk_index cannot be negative.")

    words = text.split()

    if not words:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start_word = 0

    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        chunk_words = words[start_word:end_word]
        chunk_content = " ".join(chunk_words)
        chunk_index = starting_chunk_index + len(chunks)

        chunks.append(
            TextChunk(
                chunk_id=generate_chunk_id(
                    document_id=document_id,
                    section_index=section_index,
                    chunk_index=chunk_index,
                    text=chunk_content,
                ),
                document_id=document_id,
                chunk_index=chunk_index,
                section_index=section_index,
                page_number=page_number,
                heading=heading,
                text=chunk_content,
                start_word=start_word,
                end_word=end_word,
                word_count=len(chunk_words),
            )
        )

        if end_word == len(words):
            break

        start_word += step

    return chunks
