import { apiClient, ApiError, type ApiResult } from "../../lib/api/client";
import { hasSessionApiKey } from "../../lib/api/client";

export type LivenessResponse = {
  status: "alive";
};

export type ComponentHealth = {
  status: "healthy" | "unhealthy" | "not_configured";
  detail?: string | null;
};

export type ReadinessResponse = {
  status: "ready" | "not_ready";
  components: Record<string, ComponentHealth>;
};

export const healthPaths = {
  live: "/api/v1/health/live",
  ready: "/api/v1/health/ready",
} as const;

export function getApiBaseUrl() {
  return import.meta.env.VITE_API_BASE_URL ?? "/api/v1";
}

export function isSessionApiKeySet() {
  return hasSessionApiKey();
}

export function fetchLiveness(): Promise<ApiResult<LivenessResponse>> {
  return apiClient.json<LivenessResponse>(healthPaths.live);
}

function isReadinessResponse(value: unknown): value is ReadinessResponse {
  return Boolean(
    value && typeof value === "object" && "status" in value && "components" in value,
  );
}

export async function fetchReadiness(): Promise<ApiResult<ReadinessResponse>> {
  try {
    return await apiClient.json<ReadinessResponse>(healthPaths.ready);
  } catch (error) {
    if (
      error instanceof ApiError &&
      error.status === 503 &&
      isReadinessResponse(error.body)
    ) {
      return { data: error.body, requestId: error.requestId };
    }
    throw error;
  }
}
