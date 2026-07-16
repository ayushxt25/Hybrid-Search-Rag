import { useQuery } from "@tanstack/react-query";

import { apiClient } from "../../lib/api/client";

export type LivenessResponse = {
  status: "alive";
};

export function useHealth() {
  return useQuery({
    queryKey: ["health", "live"],
    queryFn: () => apiClient.json<LivenessResponse>("/api/v1/health/live"),
    retry: 1,
    retryDelay: 250,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  });
}
