import { describe, expect, it, vi } from "vitest";

import { ApiClient, ApiError } from "./client";

describe("ApiClient", () => {
  it("uses a relative API base URL by default", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({ status: "alive" }),
    });
    const client = new ApiClient({ fetcher });

    await client.json("/api/v1/health/live");

    expect(fetcher).toHaveBeenCalledWith(
      "/api/v1/health/live",
      expect.objectContaining({ headers: expect.any(Headers) }),
    );
  });

  it("parses request IDs from response headers", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({
        "content-type": "application/json",
        "X-Request-ID": "rid-1",
      }),
      json: async () => ({ status: "alive" }),
    });
    const client = new ApiClient({ baseUrl: "http://test", fetcher });

    const response = await client.json<{ status: string }>("/health");

    expect(response.requestId).toBe("rid-1");
    expect(response.data.status).toBe("alive");
  });

  it("handles non-json errors safely", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      headers: new Headers({ "content-type": "text/plain", "X-Request-ID": "rid-2" }),
      json: async () => {
        throw new Error("should not parse");
      },
    });
    const client = new ApiClient({ baseUrl: "http://test", fetcher });

    await expect(client.json("/broken")).rejects.toMatchObject({
      status: 500,
      requestId: "rid-2",
      detail: "Request failed.",
    });
  });

  it("times out bounded requests", async () => {
    const client = new ApiClient({
      baseUrl: "http://test",
      timeoutMs: 1,
      fetcher: (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
    });

    await expect(client.json("/slow")).rejects.toMatchObject({
      detail: "timeout",
    });
    await expect(client.json("/slow")).rejects.toBeInstanceOf(ApiError);
  });
});
