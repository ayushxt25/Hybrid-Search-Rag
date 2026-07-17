import { FileText, RefreshCw, Trash2 } from "lucide-react";

import { EmptyState, Skeleton } from "../../../components/feedback/States";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { IconButton } from "../../../components/ui/IconButton";
import type { IndexedDocumentSummary } from "../types";
import { contentTypeLabel, formatDate, truncateId } from "../utils";

type DocumentListProps = {
  documents: IndexedDocumentSummary[];
  isLoading: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
  onInspect: (documentId: string) => void;
  onDelete: (document: IndexedDocumentSummary) => void;
};

export function DocumentList({
  documents,
  isLoading,
  isRefreshing,
  onRefresh,
  onInspect,
  onDelete,
}: DocumentListProps) {
  if (isLoading) {
    return (
      <Card>
        <Skeleton className="h-6 w-48" />
        <div className="mt-5 space-y-3">
          <Skeleton className="h-14" />
          <Skeleton className="h-14" />
          <Skeleton className="h-14" />
        </div>
      </Card>
    );
  }

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={FileText}
        title="No indexed documents"
        description="Upload a supported document to populate the index."
      />
    );
  }

  return (
    <Card>
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="font-semibold">Indexed documents</h2>
          <p className="text-sm text-muted">{documents.length} returned</p>
        </div>
        <IconButton
          label="Refresh documents"
          onClick={onRefresh}
          disabled={isRefreshing}
        >
          <RefreshCw className="h-4 w-4" />
        </IconButton>
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border text-xs uppercase text-muted">
            <tr>
              <th className="py-3 pr-4">Filename</th>
              <th className="py-3 pr-4">Type</th>
              <th className="py-3 pr-4">Document ID</th>
              <th className="py-3 pr-4">Chunks</th>
              <th className="py-3 pr-4">Indexed</th>
              <th className="py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((document) => (
              <tr key={document.document_id} className="border-b border-border/70">
                <td className="max-w-56 truncate py-3 pr-4">
                  {document.filename ?? "Not returned"}
                </td>
                <td className="py-3 pr-4">
                  <Badge>{contentTypeLabel(document.content_type)}</Badge>
                </td>
                <td className="py-3 pr-4 font-mono text-xs text-muted">
                  {truncateId(document.document_id)}
                </td>
                <td className="py-3 pr-4">{document.chunk_count}</td>
                <td className="py-3 pr-4">{formatDate(document.indexed_at)}</td>
                <td className="py-3">
                  <div className="flex justify-end gap-2">
                    <Button onClick={() => onInspect(document.document_id)}>
                      Inspect
                    </Button>
                    <IconButton
                      label={`Delete ${document.filename ?? "document"}`}
                      onClick={() => onDelete(document)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </IconButton>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid gap-3 md:hidden">
        {documents.map((document) => (
          <div
            key={document.document_id}
            className="rounded-token border border-border bg-background p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h3 className="truncate font-medium">
                  {document.filename ?? "Not returned"}
                </h3>
                <p className="mt-1 font-mono text-xs text-muted">
                  {truncateId(document.document_id)}
                </p>
              </div>
              <Badge>{document.chunk_count} chunks</Badge>
            </div>
            <p className="mt-3 text-sm text-muted">
              {contentTypeLabel(document.content_type)}
            </p>
            <div className="mt-4 flex gap-2">
              <Button onClick={() => onInspect(document.document_id)}>Inspect</Button>
              <Button variant="ghost" onClick={() => onDelete(document)}>
                Delete
              </Button>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
