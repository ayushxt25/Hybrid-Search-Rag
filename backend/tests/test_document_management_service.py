from pathlib import Path
from unittest.mock import Mock

import pytest

from app.documents.service import DocumentManagementService, validate_document_id
from app.schemas.documents import IndexedDocumentDetail, IndexedDocumentSummary
from app.schemas.indexing import IndexedDocumentResult

DOCUMENT_ID = "a" * 64
OTHER_DOCUMENT_ID = "b" * 64
CONTENT_HASH = "c" * 64


def test_validate_document_id_rejects_invalid_id() -> None:
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        validate_document_id("bad-id")


def test_list_returns_vector_store_documents() -> None:
    vector_store = Mock()
    summary = IndexedDocumentSummary(
        document_id=DOCUMENT_ID,
        filename="policy.txt",
        content_type=None,
        content_hash=CONTENT_HASH,
        chunk_count=2,
        indexed_at=None,
    )
    vector_store.list_documents.return_value = ([summary], None)
    service = DocumentManagementService(vector_store=vector_store)

    response = service.list_documents(limit=20)

    assert response.documents == [summary]
    assert response.next_cursor is None
    vector_store.list_documents.assert_called_once_with(limit=20, cursor=None)


def test_detail_returns_none_when_absent() -> None:
    vector_store = Mock()
    vector_store.get_document.return_value = None
    service = DocumentManagementService(vector_store=vector_store)

    assert service.get_document(DOCUMENT_ID) is None


def test_delete_missing_document_returns_none() -> None:
    vector_store = Mock()
    vector_store.delete_document.return_value = 0
    service = DocumentManagementService(vector_store=vector_store)

    assert service.delete_document(DOCUMENT_ID) is None


def test_delete_document_returns_counted_response() -> None:
    vector_store = Mock()
    vector_store.delete_document.return_value = 3
    service = DocumentManagementService(vector_store=vector_store)

    response = service.delete_document(DOCUMENT_ID)

    assert response is not None
    assert response.document_id == DOCUMENT_ID
    assert response.deleted_chunks == 3
    assert response.deleted is True


def test_replacement_preserves_old_document_when_indexing_fails(tmp_path: Path) -> None:
    vector_store = Mock()
    service = DocumentManagementService(vector_store=vector_store)

    with pytest.raises(RuntimeError, match="parse failed"):
        service.replace_document(
            document_path=tmp_path / "policy.txt",
            replace_document_id=DOCUMENT_ID,
            index_document=Mock(side_effect=RuntimeError("parse failed")),
        )

    vector_store.delete_document.assert_not_called()


def test_successful_replacement_uses_supplied_indexer(tmp_path: Path) -> None:
    vector_store = Mock()
    service = DocumentManagementService(vector_store=vector_store)
    result = IndexedDocumentResult(
        document_id=OTHER_DOCUMENT_ID,
        content_hash=OTHER_DOCUMENT_ID,
        file_name="policy.txt",
        file_extension=".txt",
        chunk_count=1,
        indexed_points=1,
    )
    index_document = Mock(return_value=result)

    response = service.replace_document(
        document_path=tmp_path / "policy.txt",
        replace_document_id=DOCUMENT_ID,
        index_document=index_document,
    )

    assert response == result
    index_document.assert_called_once()


def test_detail_metadata_is_safe_to_return() -> None:
    detail = IndexedDocumentDetail(
        document_id=DOCUMENT_ID,
        filename="policy.txt",
        content_type=None,
        content_hash=CONTENT_HASH,
        chunk_count=2,
        indexed_at=None,
        chunk_indices=[0, 1],
        page_numbers=[2],
        headings=["Benefits"],
    )

    assert "text" not in detail.model_dump()
    assert "vector" not in detail.model_dump()
