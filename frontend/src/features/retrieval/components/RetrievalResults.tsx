import {
  EmptyState,
  ErrorState,
  LoadingState,
} from "../../../components/feedback/States";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import type { IndexedDocumentSummary } from "../../documents/types";
import type { RetrievalMode, SearchResponse } from "../types";
import { RetrievalResultCard } from "./RetrievalResultCard";

type RetrievalResultsProps = {
  response: SearchResponse | null;
  mode: RetrievalMode;
  isSearching: boolean;
  errorMessage: string | null;
  documentFilterCount: number;
  contentTypeFilterCount: number;
  diagnosticsEnabled: boolean;
  durationMs: number | null;
  documents: IndexedDocumentSummary[];
  onRetry: () => void;
};

export function RetrievalResults({
  response,
  mode,
  isSearching,
  errorMessage,
  documentFilterCount,
  contentTypeFilterCount,
  diagnosticsEnabled,
  durationMs,
  documents,
  onRetry,
}: RetrievalResultsProps) {
  if (!response && isSearching) {
    return <LoadingState label="Running retrieval search..." />;
  }

  if (!response && errorMessage) {
    return <ErrorState title="Search failed" description={errorMessage} />;
  }

  if (!response) {
    return (
      <EmptyState
        title="No retrieval request yet"
        description="Submit a query to inspect ranked chunks and optional score diagnostics."
      />
    );
  }

  return (
    <div className="space-y-4" aria-live="polite">
      <div className="rounded-token border border-border bg-surface p-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="accent">{mode}</Badge>
          <Badge tone="neutral">{response.result_count} returned</Badge>
          <Badge tone="neutral">{documentFilterCount} document filters</Badge>
          <Badge tone="neutral">{contentTypeFilterCount} content-type filters</Badge>
          <Badge tone={diagnosticsEnabled ? "success" : "neutral"}>
            diagnostics {diagnosticsEnabled ? "enabled" : "disabled"}
          </Badge>
          {durationMs !== null && (
            <Badge tone="neutral">{durationMs} ms client-observed duration</Badge>
          )}
        </div>
        {isSearching && (
          <p className="mt-3 text-sm text-muted">
            Refreshing results. Previous successful response remains visible.
          </p>
        )}
        {errorMessage && (
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-danger">
            <span>{errorMessage}</span>
            <Button
              type="button"
              variant="ghost"
              className="h-8 px-2"
              onClick={onRetry}
            >
              Retry
            </Button>
          </div>
        )}
      </div>

      {response.results.length === 0 ? (
        <EmptyState
          title="No matching chunks"
          description="Try a different query or remove document/content-type filters."
        />
      ) : (
        response.results.map((result, index) => (
          <RetrievalResultCard
            key={result.chunk_id}
            result={result}
            rank={index + 1}
            documents={documents}
          />
        ))
      )}
    </div>
  );
}
