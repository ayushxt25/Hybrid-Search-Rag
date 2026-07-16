import pytest
from pydantic import ValidationError

from app.retrieval import filters as filters_module
from app.retrieval.filters import (
    MAX_CONTENT_TYPE_FILTERS,
    MAX_DOCUMENT_FILTERS,
    RetrievalFilters,
)
from app.vectorstore.qdrant import build_qdrant_filter

DOCUMENT_ID = "a" * 64
OTHER_DOCUMENT_ID = "b" * 64


def test_empty_filters() -> None:
    filters = RetrievalFilters()

    assert filters.document_ids == []
    assert filters.content_types == []


def test_single_document_id() -> None:
    filters = RetrievalFilters(document_ids=[f" {DOCUMENT_ID} "])

    assert filters.document_ids == [DOCUMENT_ID]


def test_multiple_document_ids() -> None:
    filters = RetrievalFilters(document_ids=[DOCUMENT_ID, OTHER_DOCUMENT_ID])

    assert filters.document_ids == [DOCUMENT_ID, OTHER_DOCUMENT_ID]


def test_duplicate_ids_removed() -> None:
    filters = RetrievalFilters(document_ids=[DOCUMENT_ID, DOCUMENT_ID])

    assert filters.document_ids == [DOCUMENT_ID]


def test_legacy_document_id_merged_with_document_ids() -> None:
    filters = RetrievalFilters.from_legacy(
        document_id=DOCUMENT_ID,
        document_ids=[DOCUMENT_ID, OTHER_DOCUMENT_ID],
    )

    assert filters.document_ids == [DOCUMENT_ID, OTHER_DOCUMENT_ID]


def test_content_types_normalized() -> None:
    filters = RetrievalFilters(content_types=[" Text/Plain "])

    assert filters.content_types == ["text/plain"]


def test_duplicate_content_types_removed() -> None:
    filters = RetrievalFilters(content_types=["text/plain", "TEXT/PLAIN"])

    assert filters.content_types == ["text/plain"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"document_ids": [" "]},
        {"content_types": [" "]},
    ],
)
def test_blank_values_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError, match="blank"):
        RetrievalFilters(**kwargs)


def test_invalid_document_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="64-character hexadecimal"):
        RetrievalFilters(document_ids=["bad-id"])


def test_unsupported_content_types_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported content_type"):
        RetrievalFilters(content_types=["application/json"])


def test_maximum_counts_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    document_ids = [f"{index:064x}" for index in range(MAX_DOCUMENT_FILTERS + 1)]
    content_types = [
        f"application/x-test-{index}" for index in range(MAX_CONTENT_TYPE_FILTERS + 1)
    ]
    monkeypatch.setattr(
        filters_module,
        "SUPPORTED_CONTENT_TYPES",
        set(content_types),
    )

    with pytest.raises(ValidationError):
        RetrievalFilters(document_ids=document_ids)

    with pytest.raises(ValidationError):
        RetrievalFilters(content_types=content_types)


def test_qdrant_filter_for_document_ids() -> None:
    qdrant_filter = build_qdrant_filter(
        RetrievalFilters(document_ids=[DOCUMENT_ID, OTHER_DOCUMENT_ID])
    )

    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert condition.key == "document_id"
    assert condition.match.any == [DOCUMENT_ID, OTHER_DOCUMENT_ID]


def test_qdrant_filter_for_content_types() -> None:
    qdrant_filter = build_qdrant_filter(RetrievalFilters(content_types=["text/plain"]))

    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert condition.key == "content_type"
    assert condition.match.value == "text/plain"


def test_combined_filters_use_and_between_groups() -> None:
    qdrant_filter = build_qdrant_filter(
        RetrievalFilters(
            document_ids=[DOCUMENT_ID],
            content_types=["text/plain"],
        )
    )

    assert qdrant_filter is not None
    assert [condition.key for condition in qdrant_filter.must] == [
        "document_id",
        "content_type",
    ]


def test_no_filters_returns_none() -> None:
    assert build_qdrant_filter(RetrievalFilters()) is None
