import { apiClient, type ApiResult } from "../../lib/api/client";
import type {
  DocumentDeletionResponse,
  DocumentListResponse,
  IndexedDocumentDetail,
  IndexedDocumentResult,
} from "./types";

const documentsPath = "/api/v1/documents";

export function listDocuments(): Promise<ApiResult<DocumentListResponse>> {
  return apiClient.json<DocumentListResponse>(documentsPath);
}

export function getDocument(
  documentId: string,
): Promise<ApiResult<IndexedDocumentDetail>> {
  return apiClient.json<IndexedDocumentDetail>(
    `${documentsPath}/${encodeURIComponent(documentId)}`,
  );
}

export function uploadDocument(file: File): Promise<ApiResult<IndexedDocumentResult>> {
  const body = new FormData();
  body.append("file", file);
  return apiClient.multipart<IndexedDocumentResult>(`${documentsPath}/ingest`, body);
}

export function replaceDocument({
  documentId,
  file,
}: {
  documentId: string;
  file: File;
}): Promise<ApiResult<IndexedDocumentResult>> {
  const body = new FormData();
  body.append("file", file);
  body.append("replace_document_id", documentId);
  return apiClient.multipart<IndexedDocumentResult>(`${documentsPath}/ingest`, body);
}

export function deleteDocument(
  documentId: string,
): Promise<ApiResult<DocumentDeletionResponse>> {
  return apiClient.json<DocumentDeletionResponse>(
    `${documentsPath}/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}
