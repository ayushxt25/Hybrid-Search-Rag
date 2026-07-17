import { useState } from "react";

import { ErrorState } from "../components/feedback/States";
import { Toast } from "../components/feedback/Toast";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { ConfirmDialog } from "../components/ui/ConfirmDialog";
import { HealthStatusBadge } from "../features/health/HealthStatus";
import type { HealthUiState } from "../features/health/state";
import { useHealth } from "../features/health/useHealth";
import { ApiAccessPanel } from "../features/documents/components/ApiAccessPanel";
import { DocumentDetailDialog } from "../features/documents/components/DocumentDetailDialog";
import { DocumentList } from "../features/documents/components/DocumentList";
import { DocumentUploadPanel } from "../features/documents/components/DocumentUploadPanel";
import {
  useDeleteDocument,
  useDocumentDetail,
  useDocumentList,
  useReplaceDocument,
  useUploadDocument,
} from "../features/documents/hooks";
import type { IndexedDocumentSummary } from "../features/documents/types";
import { documentErrorMessage, maxUploadLabel } from "../features/documents/utils";

function documentHealthState(health: {
  isLoading: boolean;
  isError: boolean;
}): HealthUiState {
  if (health.isLoading) return "checking";
  if (health.isError) return "unavailable";
  return "connected";
}

export function DocumentsPage() {
  const health = useHealth();
  const list = useDocumentList();
  const upload = useUploadDocument();
  const replace = useReplaceDocument();
  const remove = useDeleteDocument();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<IndexedDocumentSummary | null>(
    null,
  );
  const [toast, setToast] = useState<string | null>(null);
  const detail = useDocumentDetail(selectedId);
  const backendUnavailable = health.isError;
  const backendState = documentHealthState(health);
  const backendCopy =
    backendState === "connected"
      ? "Document operations are available."
      : backendState === "checking"
        ? "Checking backend availability."
        : "Writes are disabled while the backend is unavailable.";
  const documents = list.data?.data.documents ?? [];
  const writeDisabled =
    backendUnavailable || upload.isPending || replace.isPending || remove.isPending;

  function showToast(message: string) {
    setToast(message);
    window.setTimeout(() => setToast(null), 3000);
  }

  async function handleUpload(file: File) {
    try {
      await upload.mutateAsync(file);
      showToast("Document uploaded.");
    } catch {
      /* surfaced below */
    }
  }

  async function handleReplace(file: File) {
    if (!selectedId) return;
    try {
      const response = await replace.mutateAsync({ documentId: selectedId, file });
      setSelectedId(null);
      setPendingDelete(null);
      showToast(`${response.data.file_name} replaced.`);
    } catch {
      /* surfaced below */
    }
  }

  async function handleDelete(documentId = selectedId) {
    if (!documentId) return;
    try {
      await remove.mutateAsync(documentId);
      setSelectedId(null);
      setPendingDelete(null);
      showToast("Document deleted.");
    } catch {
      /* surfaced below */
    }
  }

  const mutationError =
    upload.error ?? replace.error ?? remove.error ?? (list.error as unknown);

  return (
    <>
      <PageHeader
        title="Documents"
        description="Upload, inspect, replace, and delete indexed source documents."
      />
      <div className="fixed bottom-4 right-4 z-50">
        {toast ? <Toast message={toast} /> : null}
      </div>
      <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          <Card>
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="font-semibold">Backend connection</h2>
                <p className="mt-1 text-sm text-muted">{backendCopy}</p>
              </div>
              <HealthStatusBadge state={backendState} />
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Badge>TXT</Badge>
              <Badge>Markdown</Badge>
              <Badge>PDF</Badge>
              <Badge>DOCX</Badge>
              <Badge>Max {maxUploadLabel}</Badge>
            </div>
          </Card>
          <ApiAccessPanel onChange={() => list.refetch()} />
          <DocumentUploadPanel
            disabled={writeDisabled}
            isUploading={upload.isPending}
            error={upload.error ? documentErrorMessage(upload.error) : null}
            onUpload={handleUpload}
          />
        </div>
        <div className="space-y-6">
          {mutationError ? (
            <ErrorState
              title="Document request failed"
              description={documentErrorMessage(mutationError)}
              onRetry={() => list.refetch()}
            />
          ) : null}
          <DocumentList
            documents={documents}
            isLoading={list.isLoading}
            isRefreshing={list.isFetching}
            onRefresh={() => list.refetch()}
            onInspect={setSelectedId}
            onDelete={setPendingDelete}
          />
        </div>
      </div>
      <DocumentDetailDialog
        open={Boolean(selectedId)}
        documentId={selectedId}
        detail={detail.data?.data ?? null}
        isLoading={detail.isLoading}
        error={detail.error ? documentErrorMessage(detail.error) : null}
        isReplacing={replace.isPending}
        isDeleting={remove.isPending}
        onClose={() => {
          setSelectedId(null);
          setPendingDelete(null);
        }}
        onRetry={() => detail.refetch()}
        onReplace={handleReplace}
        onDelete={() => handleDelete(pendingDelete?.document_id ?? selectedId)}
      />
      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete document?"
        message={`Delete ${pendingDelete?.filename ?? "this document"} and remove its indexed chunks?`}
        confirmLabel="Delete"
        destructive
        isLoading={remove.isPending}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => handleDelete(pendingDelete?.document_id)}
      />
    </>
  );
}
