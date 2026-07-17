import { apiClient, type ApiResult } from "../../lib/api/client";
import type { GroundedAnswerPayload, GroundedAnswerResponse } from "./types";

export function askGroundedAnswer(
  payload: GroundedAnswerPayload,
): Promise<ApiResult<GroundedAnswerResponse>> {
  const body: GroundedAnswerPayload = {
    question: payload.question,
    limit: payload.limit,
    candidate_limit: payload.candidate_limit,
  };
  if (payload.document_ids?.length) body.document_ids = payload.document_ids;
  if (payload.content_types?.length) body.content_types = payload.content_types;

  return apiClient.json<GroundedAnswerResponse>("/api/v1/answers/grounded", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
