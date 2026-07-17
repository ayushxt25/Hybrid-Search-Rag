import { useMemo, useRef, useState } from "react";

import { PageHeader } from "../components/layout/PageHeader";
import { Section } from "../components/ui/Section";
import { ApiAccessPanel } from "../features/documents/components/ApiAccessPanel";
import { useDocumentList } from "../features/documents/hooks";
import { useHealth } from "../features/health/useHealth";
import { useRetrievalSearch } from "../features/retrieval/hooks";
import { RetrievalExplanation } from "../features/retrieval/components/RetrievalExplanation";
import { RetrievalFilters } from "../features/retrieval/components/RetrievalFilters";
import { RetrievalForm } from "../features/retrieval/components/RetrievalForm";
import { RetrievalResults } from "../features/retrieval/components/RetrievalResults";
import type {
  RetrievalMode,
  SearchPayload,
  SearchRequest,
  SearchResponse,
} from "../features/retrieval/types";
import { unique } from "../features/retrieval/validation";
import { ApiError } from "../lib/api/client";

function errorMessage(error: unknown) {
  if (!(error instanceof ApiError)) return "Search failed safely.";
  if (error.status === 0 && error.detail === "timeout") return "Request timed out.";
  if (error.status === 0) return "Network request failed.";
  if (error.status === 401 || error.status === 403) {
    return "API credentials were not accepted.";
  }
  if (error.status === 422) {
    return error.detail && error.detail !== "[object Object]"
      ? error.detail
      : "Check the query, limits, and filters.";
  }
  if (error.status === 429) return "Too many requests. Wait briefly before retrying.";
  if (error.status === 503) return "Backend retrieval service is unavailable.";
  if (error.status >= 500) return "Backend search failed.";
  return error.detail || "Search failed safely.";
}

export function RetrievalPage() {
  const health = useHealth();
  const documentsQuery = useDocumentList();
  const search = useRetrievalSearch();
  const sequence = useRef(0);
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(5);
  const [candidateLimit, setCandidateLimit] = useState(20);
  const [includeDiagnostics, setIncludeDiagnostics] = useState(false);
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const [contentTypes, setContentTypes] = useState<string[]>([]);
  const [lastRequest, setLastRequest] = useState<SearchRequest | null>(null);
  const [lastResponse, setLastResponse] = useState<SearchResponse | null>(null);
  const [lastDurationMs, setLastDurationMs] = useState<number | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const backendAvailable = health.isSuccess;
  const documents = documentsQuery.data?.data.documents ?? [];

  const filteredDocumentIds = useMemo(() => unique(documentIds), [documentIds]);
  const filteredContentTypes = useMemo(() => unique(contentTypes), [contentTypes]);

  async function submit(payload: SearchPayload) {
    const current = sequence.current + 1;
    sequence.current = current;
    const started = window.performance.now();
    const request: SearchRequest = { mode, ...payload };
    setLastRequest(request);
    setLastError(null);
    try {
      const response = await search.mutateAsync(request);
      if (sequence.current !== current) return;
      setLastResponse(response.data);
      setLastDurationMs(Math.round(window.performance.now() - started));
    } catch (error) {
      if (sequence.current !== current) return;
      setLastError(errorMessage(error));
      setLastDurationMs(Math.round(window.performance.now() - started));
    }
  }

  function reset() {
    sequence.current += 1;
    setQuery("");
    setLimit(5);
    setCandidateLimit(20);
    setIncludeDiagnostics(false);
    setDocumentIds([]);
    setContentTypes([]);
    setLastRequest(null);
    setLastResponse(null);
    setLastDurationMs(null);
    setLastError(null);
    search.reset();
  }

  return (
    <>
      <PageHeader
        title="Retrieval Playground"
        description="Run dense, sparse, and weighted-RRF hybrid searches with metadata scope and optional score diagnostics."
      />
      <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
        <div className="space-y-6">
          <RetrievalForm
            mode={mode}
            query={query}
            limit={limit}
            candidateLimit={candidateLimit}
            includeDiagnostics={includeDiagnostics}
            documentIds={filteredDocumentIds}
            contentTypes={filteredContentTypes}
            backendAvailable={backendAvailable}
            isSearching={search.isPending}
            onModeChange={setMode}
            onQueryChange={setQuery}
            onLimitChange={setLimit}
            onCandidateLimitChange={setCandidateLimit}
            onIncludeDiagnosticsChange={setIncludeDiagnostics}
            onSubmit={submit}
            onReset={reset}
          />
          <RetrievalFilters
            documents={documents}
            documentsLoading={documentsQuery.isLoading}
            selectedDocumentIds={filteredDocumentIds}
            selectedContentTypes={filteredContentTypes}
            onDocumentChange={(ids) => setDocumentIds(unique(ids))}
            onContentTypeChange={(types) => setContentTypes(unique(types))}
          />
          <ApiAccessPanel
            onChange={() => {
              health.refetch();
              documentsQuery.refetch();
            }}
          />
        </div>
        <div className="space-y-6">
          <Section
            title="Results"
            description="Public search results only; backend internals and raw payloads are not displayed."
          >
            <RetrievalResults
              response={lastResponse}
              mode={lastRequest?.mode ?? mode}
              isSearching={search.isPending}
              errorMessage={lastError}
              documentFilterCount={lastRequest?.document_ids?.length ?? 0}
              contentTypeFilterCount={lastRequest?.content_types?.length ?? 0}
              diagnosticsEnabled={lastRequest?.include_score_diagnostics ?? false}
              durationMs={lastDurationMs}
              documents={documents}
              onRetry={() => {
                if (lastRequest) submit(lastRequest);
              }}
            />
          </Section>
          <RetrievalExplanation />
        </div>
      </div>
    </>
  );
}
