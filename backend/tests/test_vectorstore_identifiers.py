from uuid import UUID

import pytest

from app.vectorstore.identifiers import generate_qdrant_point_id


def test_qdrant_point_id_is_deterministic_uuid() -> None:
    chunk_id = "a" * 64

    first_id = generate_qdrant_point_id(chunk_id)
    second_id = generate_qdrant_point_id(chunk_id)

    assert first_id == second_id
    assert str(UUID(first_id)) == first_id


def test_qdrant_point_id_changes_with_chunk_id() -> None:
    first_id = generate_qdrant_point_id("a" * 64)
    second_id = generate_qdrant_point_id("b" * 64)

    assert first_id != second_id


@pytest.mark.parametrize(
    "chunk_id",
    [
        "",
        "a" * 63,
        "a" * 65,
        "z" * 64,
    ],
)
def test_qdrant_point_id_rejects_invalid_chunk_id(
    chunk_id: str,
) -> None:
    with pytest.raises(ValueError, match="chunk_id must"):
        generate_qdrant_point_id(chunk_id)
