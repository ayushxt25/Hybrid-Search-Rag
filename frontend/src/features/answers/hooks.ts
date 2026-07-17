import { useMutation } from "@tanstack/react-query";

import { askGroundedAnswer } from "./api";

export function useGroundedAnswer() {
  return useMutation({
    mutationFn: askGroundedAnswer,
    retry: false,
  });
}
