export type RetrievalMode = "dense" | "sparse" | "hybrid";

export type BranchScoreDiagnostic = {
  raw_score: number | null;
  rank: number | null;
  weight: number;
  rrf_contribution: number;
};

export type RetrievalScoreDiagnostic = {
  dense: BranchScoreDiagnostic;
  sparse: BranchScoreDiagnostic;
  fused_score: number;
  fused_rank: number;
};

export type SearchResult = {
  point_id?: string;
  chunk_id: string;
  document_id: string;
  score: number;
  score_diagnostics?: RetrievalScoreDiagnostic;
  file_name: string;
  file_extension: string;
  chunk_index: number;
  section_index: number;
  page_number: number | null;
  heading: string | null;
  text: string;
  start_word: number;
  end_word: number;
  word_count: number;
};

export type SearchResponse = {
  query: string;
  result_count: number;
  results: SearchResult[];
};

export type SearchPayload = {
  query: string;
  limit: number;
  candidate_limit?: number;
  document_ids?: string[];
  content_types?: string[];
  include_score_diagnostics: boolean;
};

export type SearchRequest = SearchPayload & {
  mode: RetrievalMode;
};

export const contentTypeOptions = [
  { label: "TXT", value: "text/plain" },
  { label: "Markdown", value: "text/markdown" },
  { label: "PDF", value: "application/pdf" },
  {
    label: "DOCX",
    value: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  },
] as const;

export const retrievalModeDescriptions: Record<RetrievalMode, string> = {
  dense: "Semantic similarity for meaning-oriented retrieval.",
  sparse: "Deterministic lexical retrieval for explicit terms.",
  hybrid: "Dense and sparse rankings combined using weighted reciprocal rank fusion.",
};
