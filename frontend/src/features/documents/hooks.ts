import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  deleteDocument,
  getDocument,
  listDocuments,
  replaceDocument,
  uploadDocument,
} from "./api";

export const documentKeys = {
  all: ["documents"] as const,
  list: () => [...documentKeys.all, "list"] as const,
  detail: (documentId: string) => [...documentKeys.all, "detail", documentId] as const,
};

export function useDocumentList() {
  return useQuery({
    queryKey: documentKeys.list(),
    queryFn: listDocuments,
    staleTime: 15000,
    refetchOnWindowFocus: false,
  });
}

export function useDocumentDetail(documentId: string | null) {
  return useQuery({
    queryKey: documentKeys.detail(documentId ?? ""),
    queryFn: () => getDocument(documentId ?? ""),
    enabled: Boolean(documentId),
    retry: false,
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: documentKeys.list() }),
  });
}

export function useReplaceDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: replaceDocument,
    onSuccess: async (response, variables) => {
      const oldDetailKey = documentKeys.detail(variables.documentId);
      const newDetailKey = documentKeys.detail(response.data.document_id);
      await queryClient.cancelQueries({ queryKey: oldDetailKey });
      queryClient.removeQueries({ queryKey: oldDetailKey });
      await queryClient.invalidateQueries({ queryKey: documentKeys.list() });
      if (response.data.document_id === variables.documentId) {
        await queryClient.invalidateQueries({ queryKey: newDetailKey });
      }
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: (_data, documentId) => {
      queryClient.invalidateQueries({ queryKey: documentKeys.list() });
      queryClient.removeQueries({ queryKey: documentKeys.detail(documentId) });
    },
  });
}
