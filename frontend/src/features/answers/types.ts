export type AnswerCitation = {
  source_number: number;
  chunk_id: string;
  document_id: string;
  file_name: string;
  heading: string | null;
  page_number: number | null;
};

export type GroundedAnswerResponse = {
  question: string;
  answer: string;
  model_name: string;
  citations: AnswerCitation[];
  citation_markers: number[];
  retrieved_result_count: number;
  context_source_count: number;
  context_truncated: boolean;
  insufficient_context: boolean;
  input_characters: number;
  output_characters: number;
  finish_reason: string | null;
};

export type GroundedAnswerPayload = {
  question: string;
  limit: number;
  candidate_limit: number;
  document_ids?: string[];
  content_types?: string[];
};

export type AnswerSummary = {
  documentFilterCount: number;
  contentTypeFilterCount: number;
  durationMs: number | null;
  requestId?: string;
};
