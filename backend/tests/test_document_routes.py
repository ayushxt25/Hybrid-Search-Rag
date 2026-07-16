from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_indexing_service,
    get_document_management_service,
)
from app.core.config import get_settings
from app.ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentDecodingError,
    DocumentLoadingError,
    DocumentTooLargeError,
    EmptyDocumentError,
    EncryptedDocumentError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.main import app
from app.schemas.documents import (
    DocumentDeletionResponse,
    IndexedDocumentDetail,
    IndexedDocumentListResponse,
    IndexedDocumentSummary,
)
from app.schemas.indexing import IndexedDocumentResult
from app.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)

client = TestClient(app)

DOCUMENT_ID = "a" * 64
CONTENT_HASH = "b" * 64


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()


def override_indexing_service(
    service: Mock,
) -> None:
    app.dependency_overrides[get_document_indexing_service] = lambda: service


def override_management_service(
    service: Mock,
) -> None:
    app.dependency_overrides[get_document_management_service] = lambda: service


def create_success_result(
    *,
    file_name: str,
    file_extension: str,
    chunk_count: int = 2,
) -> IndexedDocumentResult:
    return IndexedDocumentResult(
        document_id=DOCUMENT_ID,
        content_hash=CONTENT_HASH,
        file_name=file_name,
        file_extension=file_extension,
        chunk_count=chunk_count,
        indexed_points=chunk_count,
    )


def test_ingest_document_indexes_valid_text_file() -> None:
    service = Mock()

    uploaded_contents = (
        b"Remote Work Policy\n\nEmployees may work remotely for three days per week."
    )

    def index_document(
        document_path: Path,
    ) -> IndexedDocumentResult:
        assert document_path.exists()
        assert document_path.name == "remote_policy.txt"
        assert document_path.read_bytes() == uploaded_contents

        return create_success_result(
            file_name="remote_policy.txt",
            file_extension=".txt",
        )

    service.index_document.side_effect = index_document
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "remote_policy.txt",
                uploaded_contents,
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": DOCUMENT_ID,
        "content_hash": CONTENT_HASH,
        "file_name": "remote_policy.txt",
        "file_extension": ".txt",
        "chunk_count": 2,
        "indexed_points": 2,
    }

    service.index_document.assert_called_once()


def test_ingest_document_indexes_markdown_file() -> None:
    service = Mock()
    service.index_document.return_value = create_success_result(
        file_name="guide.md",
        file_extension=".md",
        chunk_count=1,
    )
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "guide.md",
                b"# Guide\n\nRun the health endpoint.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["file_extension"] == ".md"
    assert response.json()["indexed_points"] == 1


def test_ingest_document_preserves_pdf_filename() -> None:
    service = Mock()

    def index_document(
        document_path: Path,
    ) -> IndexedDocumentResult:
        assert document_path.name == "employee_handbook.pdf"

        return create_success_result(
            file_name="employee_handbook.pdf",
            file_extension=".pdf",
        )

    service.index_document.side_effect = index_document
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "employee_handbook.pdf",
                b"%PDF-valid-test-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["file_name"] == ("employee_handbook.pdf")


def test_ingest_document_preserves_docx_filename() -> None:
    service = Mock()

    def index_document(
        document_path: Path,
    ) -> IndexedDocumentResult:
        assert document_path.name == "employee_handbook.docx"

        return create_success_result(
            file_name="employee_handbook.docx",
            file_extension=".docx",
        )

    service.index_document.side_effect = index_document
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "employee_handbook.docx",
                b"fake-docx-content",
                (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["file_extension"] == ".docx"


def test_ingest_document_rejects_unsupported_file_type() -> None:
    service = Mock()
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "employees.csv",
                b"name,department",
                "text/csv",
            )
        },
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": ("Only .txt, .md, .pdf and .docx files are currently supported.")
    }

    service.index_document.assert_not_called()


def test_ingest_document_rejects_empty_file() -> None:
    service = Mock()
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Uploaded document is empty."}

    service.index_document.assert_not_called()


def test_ingest_document_requires_file() -> None:
    service = Mock()
    override_indexing_service(service)

    response = client.post("/api/v1/documents/ingest")

    assert response.status_code == 422
    service.index_document.assert_not_called()


def test_ingest_document_rejects_oversized_file() -> None:
    service = Mock()
    override_indexing_service(service)

    oversized_contents = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "large.txt",
                oversized_contents,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": ("Uploaded document exceeds the configured size limit.")
    }

    service.index_document.assert_not_called()


def test_ingest_document_accepts_file_below_upload_limit() -> None:
    service = Mock()
    service.index_document.return_value = create_success_result(
        file_name="small.txt",
        file_extension=".txt",
    )
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("small.txt", b"x" * 1024, "text/plain")},
    )

    assert response.status_code == 200
    service.index_document.assert_called_once()


def test_ingest_document_accepts_file_exactly_at_upload_limit() -> None:
    service = Mock()
    service.index_document.return_value = create_success_result(
        file_name="exact.txt",
        file_extension=".txt",
    )
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("exact.txt", b"x" * (10 * 1024 * 1024), "text/plain")},
    )

    assert response.status_code == 200
    service.index_document.assert_called_once()


def test_ingest_document_oversized_file_logs_safely(caplog) -> None:
    service = Mock()
    override_indexing_service(service)

    with caplog.at_level("WARNING", logger="app.security"):
        response = client.post(
            "/api/v1/documents/ingest",
            files={
                "file": (
                    "secret_filename.txt",
                    b"SECRET_CONTENT" * (1024 * 1024),
                    "text/plain",
                )
            },
        )

    assert response.status_code == 413
    assert "secret_filename.txt" not in caplog.text
    assert "SECRET_CONTENT" not in caplog.text
    service.index_document.assert_not_called()


def test_ingest_document_file_pointer_reset_before_processing() -> None:
    service = Mock()

    def index_document(document_path: Path) -> IndexedDocumentResult:
        assert document_path.read_bytes() == b"Valid uploaded contents."
        return create_success_result(file_name="policy.txt", file_extension=".txt")

    service.index_document.side_effect = index_document
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("policy.txt", b"Valid uploaded contents.", "text/plain")},
    )

    assert response.status_code == 200
    service.index_document.assert_called_once()


@pytest.mark.parametrize(
    (
        "raised_error",
        "expected_status",
        "expected_detail",
    ),
    [
        (
            UnsupportedFileTypeError("Unsupported document type."),
            415,
            "Unsupported document type.",
        ),
        (
            DocumentTooLargeError("Document exceeds loader size limit."),
            413,
            "Document exceeds loader size limit.",
        ),
        (
            EmptyDocumentError("Document contains no content."),
            422,
            "Document contains no content.",
        ),
        (
            DocumentDecodingError("Document must use UTF-8 encoding."),
            422,
            "Document must use UTF-8 encoding.",
        ),
        (
            EncryptedDocumentError("Encrypted PDFs are not supported."),
            422,
            "Encrypted PDFs are not supported.",
        ),
        (
            NoExtractableTextError("PDF contains no extractable text."),
            422,
            "PDF contains no extractable text.",
        ),
        (
            CorruptedDocumentError("Document is corrupted."),
            422,
            "Document is corrupted.",
        ),
        (
            DocumentLoadingError("Unable to load document."),
            400,
            "Unable to load document.",
        ),
    ],
)
def test_ingest_document_maps_ingestion_errors(
    raised_error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    service = Mock()
    service.index_document.side_effect = raised_error
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "policy.txt",
                b"Valid uploaded contents.",
                "text/plain",
            )
        },
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_ingest_document_maps_vector_connection_error() -> None:
    service = Mock()
    service.index_document.side_effect = VectorStoreConnectionError(
        "Qdrant is unavailable."
    )
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "policy.txt",
                b"Valid uploaded contents.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": ("The vector database is currently unavailable.")
    }


@pytest.mark.parametrize(
    "raised_error",
    [
        VectorStoreConfigurationError("Invalid collection configuration."),
        VectorStoreDataError("Invalid vector-store payload."),
    ],
)
def test_ingest_document_maps_vector_store_errors(
    raised_error: Exception,
) -> None:
    service = Mock()
    service.index_document.side_effect = raised_error
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "policy.txt",
                b"Valid uploaded contents.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": ("Document indexing failed due to a vector-store error.")
    }


def test_ingest_document_maps_runtime_error() -> None:
    service = Mock()
    service.index_document.side_effect = RuntimeError("Indexed point count mismatch.")
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "policy.txt",
                b"Valid uploaded contents.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": ("Document indexing did not complete successfully.")
    }


def test_temporary_file_is_removed_after_success() -> None:
    service = Mock()
    captured_path: Path | None = None

    def index_document(
        document_path: Path,
    ) -> IndexedDocumentResult:
        nonlocal captured_path

        captured_path = document_path
        assert document_path.exists()

        return create_success_result(
            file_name="policy.txt",
            file_extension=".txt",
        )

    service.index_document.side_effect = index_document
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "policy.txt",
                b"Valid uploaded contents.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert captured_path is not None
    assert not captured_path.exists()


def test_temporary_file_is_removed_after_failure() -> None:
    service = Mock()
    captured_path: Path | None = None

    def index_document(
        document_path: Path,
    ) -> IndexedDocumentResult:
        nonlocal captured_path

        captured_path = document_path
        assert document_path.exists()

        raise DocumentLoadingError("Unable to load document.")

    service.index_document.side_effect = index_document
    override_indexing_service(service)

    response = client.post(
        "/api/v1/documents/ingest",
        files={
            "file": (
                "policy.txt",
                b"Valid uploaded contents.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert captured_path is not None
    assert not captured_path.exists()


def test_ingest_document_rejects_missing_api_key_when_enabled() -> None:
    service = Mock()
    override_indexing_service(service)

    with patch(
        "app.api.dependencies.get_settings",
        return_value=type(
            "Settings",
            (),
            {
                "api_auth_enabled": True,
                "api_auth_key_sha256": "0" * 64,
                "api_auth_header_name": "X-API-Key",
                "observability_enabled": True,
            },
        )(),
    ):
        get_settings.cache_clear()
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("policy.txt", b"Valid uploaded contents.", "text/plain")},
        )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    service.index_document.assert_not_called()


def test_list_documents_returns_indexed_documents() -> None:
    service = Mock()
    service.list_documents.return_value = IndexedDocumentListResponse(
        documents=[
            IndexedDocumentSummary(
                document_id=DOCUMENT_ID,
                filename="policy.txt",
                content_type=None,
                content_hash=CONTENT_HASH,
                chunk_count=2,
                indexed_at=None,
            )
        ],
        next_cursor=None,
    )
    override_management_service(service)

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["document_id"] == DOCUMENT_ID
    assert "text" not in response.text
    service.list_documents.assert_called_once_with(limit=20, cursor=None)


def test_get_document_returns_detail() -> None:
    service = Mock()
    service.get_document.return_value = IndexedDocumentDetail(
        document_id=DOCUMENT_ID,
        filename="policy.txt",
        content_type=None,
        content_hash=CONTENT_HASH,
        chunk_count=2,
        indexed_at=None,
        chunk_indices=[0, 1],
        page_numbers=[3],
        headings=["Policy"],
    )
    override_management_service(service)

    response = client.get(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json()["chunk_indices"] == [0, 1]
    assert response.json()["page_numbers"] == [3]
    assert response.json()["headings"] == ["Policy"]


def test_get_document_returns_404_when_missing() -> None:
    service = Mock()
    service.get_document.return_value = None
    override_management_service(service)

    response = client.get(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Indexed document was not found."}


def test_delete_document_returns_deletion_response() -> None:
    service = Mock()
    service.delete_document.return_value = DocumentDeletionResponse(
        document_id=DOCUMENT_ID,
        deleted_chunks=2,
        deleted=True,
    )
    override_management_service(service)

    response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": DOCUMENT_ID,
        "deleted_chunks": 2,
        "deleted": True,
    }


def test_delete_document_returns_404_when_missing() -> None:
    service = Mock()
    service.delete_document.return_value = None
    override_management_service(service)

    response = client.delete(f"/api/v1/documents/{DOCUMENT_ID}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Indexed document was not found."}


def test_get_document_invalid_id_returns_422() -> None:
    service = Mock()
    service.get_document.side_effect = ValueError(
        "document_id must be a 64-character hexadecimal string."
    )
    override_management_service(service)

    response = client.get("/api/v1/documents/bad-id")

    assert response.status_code == 422


def test_list_documents_requires_search_auth_when_enabled() -> None:
    service = Mock()
    override_management_service(service)

    with patch(
        "app.api.dependencies.get_settings",
        return_value=type(
            "Settings",
            (),
            {
                "api_auth_enabled": True,
                "api_auth_key_sha256": "0" * 64,
                "api_auth_header_name": "X-API-Key",
                "api_auth_protect_search": True,
                "observability_enabled": True,
            },
        )(),
    ):
        get_settings.cache_clear()
        response = client.get("/api/v1/documents")

    assert response.status_code == 401
    service.list_documents.assert_not_called()


def test_ingest_document_replaces_when_replace_document_id_is_supplied() -> None:
    indexing_service = Mock()
    management_service = Mock()
    result = create_success_result(file_name="policy.txt", file_extension=".txt")
    management_service.replace_document.return_value = result
    override_indexing_service(indexing_service)
    override_management_service(management_service)

    response = client.post(
        "/api/v1/documents/ingest",
        data={"replace_document_id": DOCUMENT_ID},
        files={"file": ("policy.txt", b"Valid uploaded contents.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == DOCUMENT_ID
    indexing_service.index_document.assert_not_called()
    management_service.replace_document.assert_called_once()
