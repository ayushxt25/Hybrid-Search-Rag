from uuid import UUID, uuid5

QDRANT_POINT_NAMESPACE = UUID("e57034f7-68d2-48df-91be-b18ff4dc54e2")


def generate_qdrant_point_id(chunk_id: str) -> str:
    """Convert a deterministic chunk hash into a deterministic UUID."""
    if len(chunk_id) != 64:
        raise ValueError("chunk_id must be a 64-character SHA-256 hexadecimal string.")

    try:
        int(chunk_id, 16)
    except ValueError as error:
        raise ValueError(
            "chunk_id must contain only hexadecimal characters."
        ) from error

    return str(uuid5(QDRANT_POINT_NAMESPACE, chunk_id))
