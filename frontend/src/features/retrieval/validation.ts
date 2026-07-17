import type { RetrievalMode, SearchPayload } from "./types";

export const retrievalLimits = {
  minLimit: 1,
  maxLimit: 50,
  minCandidateLimit: 1,
  maxCandidateLimit: 100,
  maxDocumentFilters: 20,
  maxContentTypeFilters: 10,
};

export function unique(values: string[]) {
  return [...new Set(values.filter(Boolean))];
}

export function validateSearch(
  mode: RetrievalMode,
  payload: SearchPayload,
): string | null {
  if (!payload.query.trim()) return "Enter a retrieval query.";
  if (
    payload.limit < retrievalLimits.minLimit ||
    payload.limit > retrievalLimits.maxLimit
  ) {
    return "Result limit must be between 1 and 50.";
  }
  if ((payload.document_ids?.length ?? 0) > retrievalLimits.maxDocumentFilters) {
    return "Select no more than 20 documents.";
  }
  if ((payload.content_types?.length ?? 0) > retrievalLimits.maxContentTypeFilters) {
    return "Select no more than 10 content types.";
  }
  if (mode === "hybrid") {
    const candidateLimit = payload.candidate_limit ?? 20;
    if (
      candidateLimit < retrievalLimits.minCandidateLimit ||
      candidateLimit > retrievalLimits.maxCandidateLimit
    ) {
      return "Candidate limit must be between 1 and 100.";
    }
    if (candidateLimit < payload.limit) {
      return "Candidate limit must be greater than or equal to result limit.";
    }
  }
  return null;
}
