import { Copy, RefreshCw } from "lucide-react";
import { useRef, useState } from "react";

import { ErrorState, LoadingState } from "../../../components/feedback/States";
import { Button } from "../../../components/ui/Button";
import { ConfirmDialog } from "../../../components/ui/ConfirmDialog";
import { Dialog } from "../../../components/ui/Dialog";
import { IconButton } from "../../../components/ui/IconButton";
import { MetadataList } from "../../../components/ui/MetadataList";
import {
  acceptAttribute,
  contentTypeLabel,
  formatDate,
  validateUploadFile,
} from "../utils";
import type { IndexedDocumentDetail, IndexedDocumentSummary } from "../types";

type DocumentDetailDialogProps = {
  open: boolean;
  documentId: string | null;
  detail: IndexedDocumentDetail | null;
  isLoading: boolean;
  error: string | null;
  isReplacing: boolean;
  isDeleting: boolean;
  onClose: () => void;
  onRetry: () => void;
  onReplace: (file: File) => void;
  onDelete: () => void;
};

export function DocumentDetailDialog({
  open,
  documentId,
  detail,
  isLoading,
  error,
  isReplacing,
  isDeleting,
  onClose,
  onRetry,
  onReplace,
  onDelete,
}: DocumentDetailDialogProps) {
  const replaceInputRef = useRef<HTMLInputElement>(null);
  const [replaceFile, setReplaceFile] = useState<File | null>(null);
  const [replaceError, setReplaceError] = useState<string | null>(null);
  const [confirmReplace, setConfirmReplace] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [copied, setCopied] = useState(false);
  const title = detail?.filename ?? "Document detail";

  function chooseReplacement(file: File | null) {
    setReplaceFile(file);
    setReplaceError(validateUploadFile(file));
  }

  function requestReplace() {
    const nextError = validateUploadFile(replaceFile);
    setReplaceError(nextError);
    if (!replaceFile || nextError) return;
    setConfirmReplace(true);
  }

  async function copyId() {
    if (!documentId) return;
    const clipboard = window.navigator.clipboard;
    if (clipboard?.writeText) {
      await clipboard.writeText(documentId);
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  const safeDetail = detail as (IndexedDocumentSummary & IndexedDocumentDetail) | null;

  return (
    <>
      <Dialog open={open} title={title} onClose={onClose}>
        {isLoading ? <LoadingState label="Loading document metadata" /> : null}
        {error ? (
          <ErrorState
            title="Document unavailable"
            description={error}
            onRetry={onRetry}
          />
        ) : null}
        {safeDetail ? (
          <div className="space-y-5">
            <MetadataList
              items={[
                { label: "Filename", value: safeDetail.filename ?? "Not returned" },
                {
                  label: "Content type",
                  value: contentTypeLabel(safeDetail.content_type),
                },
                { label: "Chunks", value: safeDetail.chunk_count },
                {
                  label: "Pages",
                  value:
                    safeDetail.page_numbers.length > 0
                      ? safeDetail.page_numbers.join(", ")
                      : "Not returned",
                },
                { label: "Indexed", value: formatDate(safeDetail.indexed_at) },
                {
                  label: "Content hash",
                  value: safeDetail.content_hash
                    ? `${safeDetail.content_hash.slice(0, 12)}...`
                    : "Not returned",
                },
              ]}
            />
            <div className="rounded-token border border-border bg-background p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-xs uppercase tracking-wide text-muted">
                    Document ID
                  </p>
                  <p className="mt-1 break-all font-mono text-xs text-secondary">
                    {safeDetail.document_id}
                  </p>
                </div>
                <IconButton label="Copy document ID" onClick={copyId}>
                  <Copy className="h-4 w-4" />
                </IconButton>
              </div>
              <p role="status" className="mt-2 text-xs text-muted">
                {copied ? "Document ID copied." : "Full ID is shown only in detail."}
              </p>
            </div>
            <div className="rounded-token border border-border bg-background p-3">
              <p className="text-sm font-medium">Headings</p>
              <p className="mt-2 text-sm text-muted">
                {safeDetail.headings.length > 0
                  ? safeDetail.headings.slice(0, 8).join(", ")
                  : "No heading metadata returned."}
              </p>
            </div>
            <div className="rounded-token border border-warning/30 bg-warning/10 p-3 text-sm text-secondary">
              Replacement parses the new file before removing old indexed chunks, but it
              is not a database transaction.
            </div>
            <div>
              <label htmlFor="replacement-file" className="text-sm font-medium">
                Replacement file
              </label>
              <input
                ref={replaceInputRef}
                id="replacement-file"
                className="mt-2 block w-full text-sm text-secondary"
                type="file"
                accept={acceptAttribute}
                onChange={(event) =>
                  chooseReplacement(event.target.files?.item(0) ?? null)
                }
              />
              {replaceFile ? (
                <p className="mt-2 text-sm text-muted">{replaceFile.name}</p>
              ) : null}
              {replaceError ? (
                <p role="alert" className="mt-2 text-sm text-danger">
                  {replaceError}
                </p>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  onClick={requestReplace}
                  disabled={isReplacing || Boolean(validateUploadFile(replaceFile))}
                  isLoading={isReplacing}
                >
                  <RefreshCw className="h-4 w-4" />
                  Replace
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => setConfirmDelete(true)}
                  disabled={isDeleting}
                >
                  Delete document
                </Button>
              </div>
            </div>
          </div>
        ) : null}
      </Dialog>
      <ConfirmDialog
        open={confirmReplace}
        title="Replace document?"
        message={`Replace ${safeDetail?.filename ?? "this document"} with ${
          replaceFile?.name ?? "the selected file"
        }? The new file is parsed first, then old indexed chunks are removed. This is not transactional.`}
        confirmLabel="Replace"
        isLoading={isReplacing}
        onCancel={() => setConfirmReplace(false)}
        onConfirm={() => {
          if (replaceFile) onReplace(replaceFile);
          setConfirmReplace(false);
        }}
      />
      <ConfirmDialog
        open={confirmDelete}
        title="Delete document?"
        message={`Delete ${safeDetail?.filename ?? "this document"} and remove its indexed chunks?`}
        confirmLabel="Delete"
        destructive
        isLoading={isDeleting}
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => {
          onDelete();
          setConfirmDelete(false);
        }}
      />
    </>
  );
}
