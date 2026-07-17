import { useMutation } from "@tanstack/react-query";

import { runRetrievalSearch } from "./api";

export function useRetrievalSearch() {
  return useMutation({
    mutationFn: runRetrievalSearch,
    retry: false,
  });
}
