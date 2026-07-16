import re
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from app.schemas.documents import (
    DocumentDeletionResponse,
    IndexedDocumentDetail,
    IndexedDocumentListResponse,
)
from app.schemas.indexing import IndexedDocumentResult

DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_document_id(document_id: str) -> str:
    normalized_document_id = document_id.strip()
    if not DOCUMENT_ID_PATTERN.fullmatch(normalized_document_id):
        raise ValueError("document_id must be a 64-character hexadecimal string.")
    return normalized_document_id


class _DocumentLockRegistry:
    """Process-local keyed locks for document mutations."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[str, tuple[threading.Lock, int]] = {}

    @contextmanager
    def lock(self, document_id: str):
        lock = self._acquire_reference(document_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            self._release_reference(document_id)

    def _acquire_reference(self, document_id: str) -> threading.Lock:
        with self._registry_lock:
            lock, references = self._locks.get(
                document_id,
                (threading.Lock(), 0),
            )
            self._locks[document_id] = (lock, references + 1)
            return lock

    def _release_reference(self, document_id: str) -> None:
        with self._registry_lock:
            lock, references = self._locks[document_id]
            if references <= 1:
                del self._locks[document_id]
            else:
                self._locks[document_id] = (lock, references - 1)


class DocumentManagementService:
    def __init__(
        self,
        *,
        vector_store,
        indexing_service=None,
        lock_registry: _DocumentLockRegistry | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.indexing_service = indexing_service
        self._lock_registry = lock_registry or _DocumentLockRegistry()

    def list_documents(
        self,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> IndexedDocumentListResponse:
        if cursor is not None:
            validate_document_id(cursor)
        documents, next_cursor = self.vector_store.list_documents(
            limit=limit,
            cursor=cursor,
        )
        return IndexedDocumentListResponse(
            documents=documents,
            next_cursor=next_cursor,
        )

    def get_document(self, document_id: str) -> IndexedDocumentDetail | None:
        return self.vector_store.get_document(validate_document_id(document_id))

    def delete_document(self, document_id: str) -> DocumentDeletionResponse | None:
        normalized_document_id = validate_document_id(document_id)
        with self._lock_registry.lock(normalized_document_id):
            deleted_chunks = self.vector_store.delete_document(normalized_document_id)

        if deleted_chunks == 0:
            return None

        return DocumentDeletionResponse(
            document_id=normalized_document_id,
            deleted_chunks=deleted_chunks,
            deleted=True,
        )

    def replace_document(
        self,
        *,
        document_path: Path,
        replace_document_id: str,
        index_document: Callable[[Path], IndexedDocumentResult] | None = None,
    ) -> IndexedDocumentResult:
        normalized_document_id = validate_document_id(replace_document_id)
        if index_document is None and self.indexing_service is None:
            raise RuntimeError("Document replacement requires an indexing service.")

        index = index_document or self.indexing_service.index_document
        with self._lock_registry.lock(normalized_document_id):
            return index(document_path)
