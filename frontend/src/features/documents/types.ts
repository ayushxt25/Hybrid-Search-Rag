export type IndexedDocumentSummary = {
  document_id: string;
  filename: string | null;
  content_type: string | null;
  content_hash: string | null;
  chunk_count: number;
  indexed_at: string | null;
};

export type IndexedDocumentDetail = IndexedDocumentSummary & {
  chunk_indices: number[];
  page_numbers: number[];
  headings: string[];
};

export type DocumentListResponse = {
  documents: IndexedDocumentSummary[];
  next_cursor: string | null;
};

export type IndexedDocumentResult = {
  document_id: string;
  content_hash: string;
  file_name: string;
  file_extension: string;
  chunk_count: number;
  indexed_points: number;
};

export type DocumentDeletionResponse = {
  document_id: string;
  deleted_chunks: number;
  deleted: boolean;
};
