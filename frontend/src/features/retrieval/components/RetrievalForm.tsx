import { RotateCcw, Search } from "lucide-react";

import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Checkbox } from "../../../components/ui/Checkbox";
import { Input } from "../../../components/ui/Input";
import { Tabs } from "../../../components/ui/Tabs";
import { Textarea } from "../../../components/ui/Textarea";
import type { RetrievalMode, SearchPayload } from "../types";
import { retrievalModeDescriptions } from "../types";
import { validateSearch } from "../validation";

type RetrievalFormProps = {
  mode: RetrievalMode;
  query: string;
  limit: number;
  candidateLimit: number;
  includeDiagnostics: boolean;
  documentIds: string[];
  contentTypes: string[];
  backendAvailable: boolean;
  isSearching: boolean;
  onModeChange: (mode: RetrievalMode) => void;
  onQueryChange: (query: string) => void;
  onLimitChange: (limit: number) => void;
  onCandidateLimitChange: (limit: number) => void;
  onIncludeDiagnosticsChange: (include: boolean) => void;
  onSubmit: (payload: SearchPayload) => void;
  onReset: () => void;
};

export function RetrievalForm({
  mode,
  query,
  limit,
  candidateLimit,
  includeDiagnostics,
  documentIds,
  contentTypes,
  backendAvailable,
  isSearching,
  onModeChange,
  onQueryChange,
  onLimitChange,
  onCandidateLimitChange,
  onIncludeDiagnosticsChange,
  onSubmit,
  onReset,
}: RetrievalFormProps) {
  const payload: SearchPayload = {
    query,
    limit,
    candidate_limit: candidateLimit,
    document_ids: documentIds,
    content_types: contentTypes,
    include_score_diagnostics: includeDiagnostics,
  };
  const validationError = validateSearch(mode, payload);
  const disabled = isSearching || !backendAvailable || Boolean(validationError);

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const error = validateSearch(mode, payload);
    if (error || isSearching || !backendAvailable) return;
    onSubmit({ ...payload, query: query.trim() });
  }

  return (
    <Card>
      <form onSubmit={submit} className="space-y-5">
        <div>
          <label htmlFor="retrieval-query" className="text-sm font-medium">
            Query
          </label>
          <Textarea
            id="retrieval-query"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Ask for an explicit phrase, policy detail, or semantic concept."
            aria-describedby="retrieval-query-help retrieval-form-error"
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <p id="retrieval-query-help" className="mt-2 text-xs text-muted">
            Press Enter to search. Use Shift+Enter for a new line.
          </p>
        </div>

        <div>
          <span className="text-sm font-medium">Search mode</span>
          <div className="mt-2">
            <Tabs
              value={mode}
              onChange={(value) => onModeChange(value as RetrievalMode)}
              items={[
                { id: "dense", label: "Dense" },
                { id: "sparse", label: "Sparse" },
                { id: "hybrid", label: "Hybrid" },
              ]}
            />
          </div>
          <p className="mt-2 text-sm text-muted">{retrievalModeDescriptions[mode]}</p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="text-sm font-medium">
            Result limit
            <Input
              className="mt-2"
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(event) => onLimitChange(Number(event.target.value))}
            />
          </label>
          {mode === "hybrid" && (
            <label className="text-sm font-medium">
              Candidate limit
              <Input
                className="mt-2"
                type="number"
                min={1}
                max={100}
                value={candidateLimit}
                onChange={(event) => onCandidateLimitChange(Number(event.target.value))}
              />
            </label>
          )}
        </div>

        <label className="flex cursor-pointer items-start gap-3 rounded-token border border-border p-3 text-sm">
          <Checkbox
            checked={includeDiagnostics}
            onChange={(event) => onIncludeDiagnosticsChange(event.target.checked)}
          />
          <span>
            <span className="block font-medium">Include score diagnostics</span>
            <span className="block text-muted">
              Show branch ranks, raw scores, and weighted-RRF contributions.
            </span>
          </span>
        </label>

        <p
          id="retrieval-form-error"
          role="alert"
          className="min-h-5 text-sm text-danger"
        >
          {!backendAvailable
            ? "Backend is unavailable."
            : validationError
              ? validationError
              : ""}
        </p>

        <div className="flex flex-wrap gap-3">
          <Button
            type="submit"
            variant="primary"
            isLoading={isSearching}
            disabled={disabled}
          >
            <Search aria-hidden className="h-4 w-4" />
            Search
          </Button>
          <Button type="button" variant="secondary" onClick={onReset}>
            <RotateCcw aria-hidden className="h-4 w-4" />
            Reset
          </Button>
        </div>
      </form>
    </Card>
  );
}
