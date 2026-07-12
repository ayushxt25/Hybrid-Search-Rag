import hashlib


def generate_content_hash(content: str) -> str:
    """Generate a deterministic SHA-256 hash from normalized content."""
    if not content:
        raise ValueError("Content cannot be empty.")

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_chunk_id(
    *,
    document_id: str,
    section_index: int,
    chunk_index: int,
    text: str,
) -> str:
    """Generate a deterministic SHA-256 identifier for a chunk."""
    if not document_id:
        raise ValueError("document_id cannot be empty.")

    if section_index < 0:
        raise ValueError("section_index cannot be negative.")

    if chunk_index < 0:
        raise ValueError("chunk_index cannot be negative.")

    if not text:
        raise ValueError("Chunk text cannot be empty.")

    identifier_source = f"{document_id}:{section_index}:{chunk_index}:{text}"

    return hashlib.sha256(identifier_source.encode("utf-8")).hexdigest()
