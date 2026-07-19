import { afterEach, describe, expect, it, vi } from "vitest";

import { clearSessionApiKey, setSessionApiKey } from "../../lib/api/client";
import {
  deleteDocument,
  getDocument,
  listDocuments,
  replaceDocument,
  uploadDocument,
} from "./api";

function ok(data: unknown = {}) {
  return Promise.resolve({
    ok: true,
    status: 200,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => data,
  } as Response);
}

describe("document API", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uses the document list endpoint", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok({ documents: [] }));
    vi.stubGlobal("fetch", fetcher);
    await listDocuments();
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/documents");
  });

  it("encodes document IDs for detail requests", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok({ document_id: "abc" }));
    vi.stubGlobal("fetch", fetcher);
    await getDocument("abc/def");
    expect(fetcher.mock.calls[0][0]).toContain("/api/v1/documents/abc%2Fdef");
  });

  it("uploads the file multipart field", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    await uploadDocument(new File(["x"], "note.txt", { type: "text/plain" }));
    const body = fetcher.mock.calls[0][1].body as FormData;
    expect(body.get("file")).toBeInstanceOf(File);
  });

  it("includes replace_document_id for replacement", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    await replaceDocument({
      documentId: "a".repeat(64),
      file: new File(["x"], "note.md", { type: "text/markdown" }),
    });
    const body = fetcher.mock.calls[0][1].body as FormData;
    expect(body.get("replace_document_id")).toBe("a".repeat(64));
  });

  it("deletes by method and endpoint", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    await deleteDocument("b".repeat(64));
    expect(fetcher.mock.calls[0][0]).toContain(`/api/v1/documents/${"b".repeat(64)}`);
    expect(fetcher.mock.calls[0][1].method).toBe("DELETE");
  });

  it("uses the configured in-memory API key header", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok({ documents: [] }));
    vi.stubGlobal("fetch", fetcher);
    setSessionApiKey("secret-key");
    await listDocuments();
    expect((fetcher.mock.calls[0][1].headers as Headers).get("X-API-Key")).toBe(
      "secret-key",
    );
  });
});
