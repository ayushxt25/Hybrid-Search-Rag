import json
import logging
from datetime import UTC, datetime
from logging import LogRecord
from typing import Any

SAFE_LOG_FIELDS = {
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "timestamp",
    "reason",
    "route",
    "document_count",
    "elapsed_ms",
    "found",
    "chunk_count",
    "deleted",
    "deleted_chunk_count",
    "new_chunk_count",
    "reset_after_seconds",
    "success",
    "retrieved_result_count",
    "result_count",
    "search_type",
    "context_source_count",
    "context_truncated",
    "insufficient_context",
    "finish_reason",
    "citation_marker_count",
    "filter_document_count",
    "filter_content_type_count",
    "filtered",
    "retrieval_ms",
    "context_assembly_ms",
    "prompt_construction_ms",
    "generation_ms",
    "total_ms",
    "exception_type",
    "stage",
    "resource_type",
}


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


class SafeJsonFormatter(logging.Formatter):
    """Render log records as JSON with an explicit production-safe field list."""

    def format(self, record: LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": getattr(record, "timestamp", utc_timestamp()),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }

        for field in SAFE_LOG_FIELDS:
            if field in payload:
                continue
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
