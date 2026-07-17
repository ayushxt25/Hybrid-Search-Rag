import { useMemo, useRef, useState } from "react";

import { PageHeader } from "../components/layout/PageHeader";
import { ApiAccessPanel } from "../features/documents/components/ApiAccessPanel";
import { useDocumentList } from "../features/documents/hooks";
import { useHealth } from "../features/health/useHealth";
import { AnswerPanel } from "../features/answers/components/AnswerPanel";
import { AnswerForm } from "../features/answers/components/AnswerForm";
import { useGroundedAnswer } from "../features/answers/hooks";
import type {
  AnswerSummary,
  GroundedAnswerPayload,
  GroundedAnswerResponse,
} from "../features/answers/types";
import { RetrievalFilters } from "../features/retrieval/components/RetrievalFilters";
import { unique } from "../features/retrieval/validation";
import { ApiError } from "../lib/api/client";

function answerErrorMessage(error: unknown) {
  if (!(error instanceof ApiError)) return "Answer request failed safely.";
  if (error.status === 0 && error.detail === "timeout") return "Request timed out.";
  if (error.status === 0) return "Network request failed.";
  if (error.status === 401 || error.status === 403) {
    return "API access key may be required.";
  }
  if (error.status === 422) return "Check the question, limits, and filters.";
  if (error.status === 429) {
    return `Request limit reached.${error.retryAfter ? ` Retry after ${error.retryAfter} seconds.` : ""}`;
  }
  if (error.status === 502) {
    return "The generation provider returned an invalid response.";
  }
  if (error.status === 503) {
    return "Retrieval or generation service is temporarily unavailable.";
  }
  if (error.status >= 500) return "Grounded answer generation failed.";
  return error.detail || "Answer request failed safely.";
}

export function AnswersPage() {
  const health = useHealth();
  const documentsQuery = useDocumentList();
  const answer = useGroundedAnswer();
  const sequence = useRef(0);
  const [question, setQuestion] = useState("");
  const [limit, setLimit] = useState(5);
  const [candidateLimit, setCandidateLimit] = useState(20);
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const [contentTypes, setContentTypes] = useState<string[]>([]);
  const [response, setResponse] = useState<GroundedAnswerResponse | null>(null);
  const [summary, setSummary] = useState<AnswerSummary | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<number | null>(null);
  const backendAvailable = health.isSuccess;
  const documents = documentsQuery.data?.data.documents ?? [];
  const filteredDocumentIds = useMemo(() => unique(documentIds), [documentIds]);
  const filteredContentTypes = useMemo(() => unique(contentTypes), [contentTypes]);

  async function submit(payload: GroundedAnswerPayload) {
    const current = sequence.current + 1;
    sequence.current = current;
    const started = window.performance.now();
    setErrorMessage(null);
    try {
      const result = await answer.mutateAsync(payload);
      if (sequence.current !== current) return;
      setResponse(result.data);
      setSelectedSource(result.data.citations[0]?.source_number ?? null);
      setSummary({
        documentFilterCount: payload.document_ids?.length ?? 0,
        contentTypeFilterCount: payload.content_types?.length ?? 0,
        durationMs: Math.round(window.performance.now() - started),
        requestId: result.requestId,
      });
    } catch (error) {
      if (sequence.current !== current) return;
      setErrorMessage(answerErrorMessage(error));
      setSummary((previous) =>
        previous
          ? {
              ...previous,
              durationMs: Math.round(window.performance.now() - started),
            }
          : {
              documentFilterCount: payload.document_ids?.length ?? 0,
              contentTypeFilterCount: payload.content_types?.length ?? 0,
              durationMs: Math.round(window.performance.now() - started),
            },
      );
    }
  }

  function reset() {
    sequence.current += 1;
    setQuestion("");
    setLimit(5);
    setCandidateLimit(20);
    setDocumentIds([]);
    setContentTypes([]);
    setResponse(null);
    setSummary(null);
    setErrorMessage(null);
    setSelectedSource(null);
    answer.reset();
  }

  function selectSource(sourceNumber: number) {
    setSelectedSource(sourceNumber);
    document
      .getElementById(`source-${sourceNumber}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <>
      <PageHeader
        title="Grounded Answers"
        description="Ask questions over indexed documents and inspect validated citation sources."
      />
      <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
        <div className="space-y-6">
          <AnswerForm
            question={question}
            limit={limit}
            candidateLimit={candidateLimit}
            documentIds={filteredDocumentIds}
            contentTypes={filteredContentTypes}
            backendAvailable={backendAvailable}
            isAsking={answer.isPending}
            onQuestionChange={setQuestion}
            onLimitChange={setLimit}
            onCandidateLimitChange={setCandidateLimit}
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
        <AnswerPanel
          response={response}
          summary={summary}
          isAsking={answer.isPending}
          errorMessage={errorMessage}
          selectedSource={selectedSource}
          documents={documents}
          onSelectSource={selectSource}
        />
      </div>
    </>
  );
}
