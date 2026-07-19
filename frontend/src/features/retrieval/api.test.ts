import { afterEach, describe, expect, it, vi } from "vitest";

import { clearSessionApiKey, setSessionApiKey } from "../../lib/api/client";
import { searchDense, searchHybrid, searchSparse } from "./api";

function ok(data: unknown = { query: "q", result_count: 0, results: [] }) {
  return Promise.resolve({
    ok: true,
    status: 200,
    headers: new Headers({
      "content-type": "application/json",
      "X-Request-ID": "req-1",
    }),
    json: async () => data,
  } as Response);
}

const basePayload = {
  query: "marker",
  limit: 5,
  candidate_limit: 25,
  document_ids: ["a".repeat(64)],
  content_types: ["text/plain"],
  include_score_diagnostics: true,
};

describe("retrieval API", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("sends exact dense diagnostics true payload", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    const result = await searchDense(basePayload);
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/search/dense");
    const body = JSON.parse(fetcher.mock.calls[0][1].body as string);
    expect(body).toEqual({
      query: "marker",
      limit: 5,
      document_ids: ["a".repeat(64)],
      content_types: ["text/plain"],
      include_score_diagnostics: true,
    });
    expect(typeof body.include_score_diagnostics).toBe("boolean");
    expect(result.requestId).toBe("req-1");
  });

  it("sends exact dense diagnostics false payload and omits empty filters", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    await searchDense({
      query: "  marker  ",
      limit: 5,
      document_ids: [],
      content_types: [],
      include_score_diagnostics: false,
    });
    expect(JSON.parse(fetcher.mock.calls[0][1].body as string)).toEqual({
      query: "marker",
      limit: 5,
      include_score_diagnostics: false,
    });
  });

  it("sends exact sparse diagnostics true payload", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    await searchSparse(basePayload);
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/search/sparse");
    expect(JSON.parse(fetcher.mock.calls[0][1].body as string)).toEqual({
      query: "marker",
      limit: 5,
      document_ids: ["a".repeat(64)],
      content_types: ["text/plain"],
      include_score_diagnostics: true,
    });
  });

  it("sends exact hybrid diagnostics true payload with candidate limit", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    await searchHybrid(basePayload);
    const body = JSON.parse(fetcher.mock.calls[0][1].body as string);
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/search/hybrid");
    expect(body).toEqual({
      query: "marker",
      limit: 5,
      candidate_limit: 25,
      document_ids: ["a".repeat(64)],
      content_types: ["text/plain"],
      include_score_diagnostics: true,
    });
  });

  it("uses the in-memory API key header", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    setSessionApiKey("retrieval-key");
    await searchDense(basePayload);
    expect((fetcher.mock.calls[0][1].headers as Headers).get("X-API-Key")).toBe(
      "retrieval-key",
    );
  });

  it("surfaces non-JSON errors safely", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        headers: new Headers({ "content-type": "text/plain" }),
        json: async () => ({}),
      } as Response),
    );
    await expect(searchDense(basePayload)).rejects.toThrow("Request failed.");
  });

  it("does not retry 422 validation responses", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({ detail: "validation failed" }),
    } as Response);
    vi.stubGlobal("fetch", fetcher);
    await expect(searchDense(basePayload)).rejects.toThrow("validation failed");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
