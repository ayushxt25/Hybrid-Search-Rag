export const answerLimits = {
  minLimit: 1,
  maxLimit: 50,
  minCandidateLimit: 1,
  maxCandidateLimit: 100,
  maxDocumentFilters: 20,
  maxContentTypeFilters: 10,
};

export function validateAnswerRequest({
  question,
  limit,
  candidateLimit,
  documentIds,
  contentTypes,
}: {
  question: string;
  limit: number;
  candidateLimit: number;
  documentIds: string[];
  contentTypes: string[];
}) {
  if (!question.trim()) return "Enter a grounded question.";
  if (limit < answerLimits.minLimit || limit > answerLimits.maxLimit) {
    return "Result limit must be between 1 and 50.";
  }
  if (
    candidateLimit < answerLimits.minCandidateLimit ||
    candidateLimit > answerLimits.maxCandidateLimit
  ) {
    return "Candidate limit must be between 1 and 100.";
  }
  if (candidateLimit < limit) {
    return "Candidate limit must be greater than or equal to result limit.";
  }
  if (documentIds.length > answerLimits.maxDocumentFilters) {
    return "Select no more than 20 documents.";
  }
  if (contentTypes.length > answerLimits.maxContentTypeFilters) {
    return "Select no more than 10 content types.";
  }
  return null;
}
