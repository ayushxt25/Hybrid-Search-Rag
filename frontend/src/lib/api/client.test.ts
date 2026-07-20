import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiClient,
  ApiError,
  clearSessionApiKey,
  hasSessionApiKey,
  SESSION_API_KEY_STORAGE_KEY,
  setSessionApiKey,
  subscribeSessionApiKey,
} from "./client";

describe("ApiClient", () => {
  afterEach(() => {
    clearSessionApiKey();
    window.sessionStorage.clear();
    vi.restoreAllMocks();
  });

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

  it("stores trimmed session API keys in sessionStorage", () => {
    setSessionApiKey("  production-key  ");

    expect(window.sessionStorage.getItem(SESSION_API_KEY_STORAGE_KEY)).toBe(
      "production-key",
    );
    expect(hasSessionApiKey()).toBe(true);
  });

  it("clears session API keys when the value is empty", () => {
    setSessionApiKey("production-key");
    setSessionApiKey("   ");

    expect(window.sessionStorage.getItem(SESSION_API_KEY_STORAGE_KEY)).toBeNull();
    expect(hasSessionApiKey()).toBe(false);
  });

  it("reads an existing sessionStorage key after initialization", async () => {
    window.sessionStorage.setItem(SESSION_API_KEY_STORAGE_KEY, "stored-key");
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({ documents: [] }),
    });

    await new ApiClient({ fetcher }).json("/api/v1/documents");

    expect((fetcher.mock.calls[0][1].headers as Headers).get("X-API-Key")).toBe(
      "stored-key",
    );
  });

  it("attaches X-API-Key only when a key exists", async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({ status: "alive" }),
    });
    const client = new ApiClient({ fetcher });

    await client.json("/api/v1/health/live");
    expect((fetcher.mock.calls[0][1].headers as Headers).has("X-API-Key")).toBe(
      false,
    );

    setSessionApiKey("production-key");
    await client.json("/api/v1/health/live");
    expect((fetcher.mock.calls[1][1].headers as Headers).get("X-API-Key")).toBe(
      "production-key",
    );
  });

  it("notifies subscribers when the session API key changes", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeSessionApiKey(listener);

    setSessionApiKey("production-key");
    clearSessionApiKey();
    unsubscribe();
    setSessionApiKey("other-key");

    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("does not crash when sessionStorage is unavailable", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(() => setSessionApiKey("production-key")).not.toThrow();
    expect(hasSessionApiKey()).toBe(true);
    expect(() => clearSessionApiKey()).not.toThrow();
    expect(hasSessionApiKey()).toBe(false);
  });
});
