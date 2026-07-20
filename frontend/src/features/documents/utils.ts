import { ApiError } from "../../lib/api/client";

export const acceptedExtensions = [".txt", ".md", ".pdf", ".docx"] as const;
export const acceptedMimeTypes = [
  "text/plain",
  "text/markdown",
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
];
export const acceptAttribute = [...acceptedExtensions, ...acceptedMimeTypes].join(",");

const defaultMaxUploadBytes = 10 * 1024 * 1024;
const configuredMaxUploadBytes = Number(import.meta.env.VITE_MAX_DOCUMENT_UPLOAD_BYTES);

export const maxUploadBytes =
  Number.isFinite(configuredMaxUploadBytes) && configuredMaxUploadBytes > 0
    ? configuredMaxUploadBytes
    : defaultMaxUploadBytes;
export const maxUploadLabel =
  maxUploadBytes % (1024 * 1024) === 0
    ? `${maxUploadBytes / (1024 * 1024)} MB`
    : `${(maxUploadBytes / (1024 * 1024)).toFixed(1)} MB`;

export function truncateId(value: string) {
  return `${value.slice(0, 10)}...${value.slice(-8)}`;
}

export function formatDate(value: string | null) {
  if (!value) return "Not returned";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function contentTypeLabel(value: string | null) {
  if (!value) return "Not returned";
  if (
    value === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ) {
    return "DOCX";
  }
  return value;
}

export function validateUploadFile(file: File | null): string | null {
  if (!file) return "Choose a document first.";
  const lowerName = file.name.toLowerCase();
  const hasAcceptedExtension = acceptedExtensions.some((extension) =>
    lowerName.endsWith(extension),
  );
  const hasAcceptedType = !file.type || acceptedMimeTypes.includes(file.type);
  if (!hasAcceptedExtension || !hasAcceptedType) {
    return "Use a TXT, Markdown, PDF, or DOCX file.";
  }
  if (file.size > maxUploadBytes) {
    return `File is larger than ${maxUploadLabel}.`;
  }
  return null;
}

export function documentErrorMessage(error: unknown) {
  if (!(error instanceof ApiError)) return "The request could not be completed.";
  if (error.detail === "timeout") return "The backend request timed out.";
  if (error.status === 401 || error.status === 403) {
    return "API credentials were not accepted. Go to System Health -> Session API key, update the key, and retry.";
  }
  if (error.status === 413) return "The selected file exceeds the backend limit.";
  if (error.status === 422)
    return error.detail || "The document could not be validated.";
  if (error.status === 503) return "The vector database is currently unavailable.";
  if (error.status === 404) return "The document was not found.";
  if (error.status >= 500) return "The backend could not complete the request.";
  return error.detail || "The request failed.";
}
