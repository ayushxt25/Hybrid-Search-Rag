import { useMutation, useQuery } from "@tanstack/react-query";

import { fetchLiveness, fetchReadiness } from "./api";

export function useHealth() {
  return useQuery({
    queryKey: ["health", "live"],
    queryFn: fetchLiveness,
    retry: 1,
    retryDelay: 250,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  });
}

export function useReadinessCheck() {
  return useMutation({
    mutationFn: fetchReadiness,
    retry: false,
  });
}
