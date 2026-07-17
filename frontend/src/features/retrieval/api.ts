import { apiClient, type ApiResult } from "../../lib/api/client";
import type { RetrievalMode, SearchPayload, SearchResponse } from "./types";

const searchPaths: Record<RetrievalMode, string> = {
  dense: "/api/v1/search/dense",
  sparse: "/api/v1/search/sparse",
  hybrid: "/api/v1/search/hybrid",
};

type SearchRequestBody = {
  query: string;
  limit: number;
  candidate_limit?: number;
  document_ids?: string[];
  content_types?: string[];
  include_score_diagnostics: boolean;
};

function payloadFor(mode: RetrievalMode, payload: SearchPayload): SearchRequestBody {
  const body: SearchRequestBody = {
    query: payload.query.trim(),
    limit: payload.limit,
    include_score_diagnostics: payload.include_score_diagnostics === true,
  };
  if (mode === "hybrid" && Number.isFinite(payload.candidate_limit)) {
    body.candidate_limit = payload.candidate_limit;
  }
  if (payload.document_ids?.length) {
    body.document_ids = payload.document_ids;
  }
  if (payload.content_types?.length) {
    body.content_types = payload.content_types;
  }
  return body;
}

export function searchDense(
  payload: SearchPayload,
): Promise<ApiResult<SearchResponse>> {
  return apiClient.json<SearchResponse>(searchPaths.dense, {
    method: "POST",
    body: JSON.stringify(payloadFor("dense", payload)),
  });
}

export function searchSparse(
  payload: SearchPayload,
): Promise<ApiResult<SearchResponse>> {
  return apiClient.json<SearchResponse>(searchPaths.sparse, {
    method: "POST",
    body: JSON.stringify(payloadFor("sparse", payload)),
  });
}

export function searchHybrid(
  payload: SearchPayload,
): Promise<ApiResult<SearchResponse>> {
  return apiClient.json<SearchResponse>(searchPaths.hybrid, {
    method: "POST",
    body: JSON.stringify(payloadFor("hybrid", payload)),
  });
}

export function runRetrievalSearch({
  mode,
  ...payload
}: SearchPayload & {
  mode: RetrievalMode;
}): Promise<ApiResult<SearchResponse>> {
  if (mode === "dense") return searchDense(payload);
  if (mode === "sparse") return searchSparse(payload);
  return searchHybrid(payload);
}
