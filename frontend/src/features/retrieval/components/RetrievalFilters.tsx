import { X } from "lucide-react";

import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Checkbox } from "../../../components/ui/Checkbox";
import { Card } from "../../../components/ui/Card";
import { contentTypeLabel, truncateId } from "../../documents/utils";
import type { IndexedDocumentSummary } from "../../documents/types";
import { contentTypeOptions } from "../types";

type RetrievalFiltersProps = {
  documents: IndexedDocumentSummary[];
  documentsLoading: boolean;
  selectedDocumentIds: string[];
  selectedContentTypes: string[];
  onDocumentChange: (ids: string[]) => void;
  onContentTypeChange: (types: string[]) => void;
};

export function RetrievalFilters({
  documents,
  documentsLoading,
  selectedDocumentIds,
  selectedContentTypes,
  onDocumentChange,
  onContentTypeChange,
}: RetrievalFiltersProps) {
  const selectedDocuments = selectedDocumentIds.map((id) => {
    const document = documents.find((item) => item.document_id === id);
    return { id, label: document?.filename ?? truncateId(id) };
  });

  function toggleDocument(documentId: string) {
    onDocumentChange(
      selectedDocumentIds.includes(documentId)
        ? selectedDocumentIds.filter((id) => id !== documentId)
        : [...selectedDocumentIds, documentId],
    );
  }

  function toggleContentType(contentType: string) {
    onContentTypeChange(
      selectedContentTypes.includes(contentType)
        ? selectedContentTypes.filter((value) => value !== contentType)
        : [...selectedContentTypes, contentType],
    );
  }

  return (
    <Card className="space-y-5">
      <div>
        <h2 className="font-semibold">Metadata filters</h2>
        <p className="mt-1 text-sm text-muted">
          Filters are sent as document IDs and exact content-type values.
        </p>
      </div>

      <section aria-labelledby="document-filter-heading">
        <div className="flex items-center justify-between gap-3">
          <h3 id="document-filter-heading" className="text-sm font-medium">
            Documents
          </h3>
          <Button
            type="button"
            variant="ghost"
            className="h-8 px-2"
            onClick={() => onDocumentChange([])}
            disabled={!selectedDocumentIds.length}
          >
            Clear
          </Button>
        </div>
        <div className="mt-2 flex flex-wrap gap-2">
          {selectedDocuments.map((document) => (
            <Badge key={document.id} tone="accent">
              {document.label}
              <button
                type="button"
                aria-label={`Remove ${document.label} filter`}
                onClick={() =>
                  onDocumentChange(
                    selectedDocumentIds.filter((id) => id !== document.id),
                  )
                }
                className="ml-1 rounded-sm text-muted hover:text-primary focus:outline-none focus:ring-2 focus:ring-focus"
              >
                <X aria-hidden className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {!selectedDocumentIds.length && (
            <span className="text-sm text-muted">No document scope selected.</span>
          )}
        </div>
        <div className="mt-3 max-h-52 space-y-2 overflow-auto rounded-token border border-border p-3">
          {documentsLoading && (
            <p className="text-sm text-muted">Loading documents...</p>
          )}
          {!documentsLoading && documents.length === 0 && (
            <p className="text-sm text-muted">No indexed documents available.</p>
          )}
          {documents.map((document) => (
            <label
              key={document.document_id}
              className="flex cursor-pointer items-start gap-3 rounded-md p-2 text-sm hover:bg-elevated"
            >
              <Checkbox
                checked={selectedDocumentIds.includes(document.document_id)}
                onChange={() => toggleDocument(document.document_id)}
              />
              <span>
                <span className="block text-primary">
                  {document.filename ?? "Untitled document"}
                </span>
                <span className="block text-xs text-muted">
                  {contentTypeLabel(document.content_type)} ·{" "}
                  {truncateId(document.document_id)}
                </span>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section aria-labelledby="content-filter-heading">
        <div className="flex items-center justify-between gap-3">
          <h3 id="content-filter-heading" className="text-sm font-medium">
            Content types
          </h3>
          <Button
            type="button"
            variant="ghost"
            className="h-8 px-2"
            onClick={() => onContentTypeChange([])}
            disabled={!selectedContentTypes.length}
          >
            Clear
          </Button>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          {contentTypeOptions.map((option) => (
            <label
              key={option.value}
              className="flex cursor-pointer items-center gap-2 rounded-token border border-border p-3 text-sm hover:bg-elevated"
            >
              <Checkbox
                checked={selectedContentTypes.includes(option.value)}
                onChange={() => toggleContentType(option.value)}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {selectedContentTypes.map((type) => (
            <Badge key={type} tone="neutral">
              {contentTypeLabel(type)}
            </Badge>
          ))}
        </div>
      </section>
    </Card>
  );
}
