import { afterEach, describe, expect, it, vi } from "vitest";

import { clearSessionApiKey } from "../../lib/api/client";
import { fetchLiveness, fetchReadiness, healthPaths } from "./api";

function json(data: unknown = {}, status = 200, headers: Record<string, string> = {}) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({
      "content-type": "application/json",
      "X-Request-ID": "rid-health",
      ...headers,
    }),
    json: async () => data,
  } as Response);
}

describe("health API", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uses the liveness endpoint", async () => {
    const fetcher = vi.fn().mockResolvedValue(await json({ status: "alive" }));
    vi.stubGlobal("fetch", fetcher);
    const result = await fetchLiveness();
    expect(fetcher.mock.calls[0][0]).toBe(`${healthPaths.live}`);
    expect(result.requestId).toBe("rid-health");
  });

  it("uses the readiness endpoint", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValue(await json({ status: "ready", components: {} }));
    vi.stubGlobal("fetch", fetcher);
    await fetchReadiness();
    expect(fetcher.mock.calls[0][0]).toBe(`${healthPaths.ready}`);
  });

  it("handles non-JSON errors safely", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        headers: new Headers({ "content-type": "text/plain" }),
        json: async () => ({}),
      } as Response),
    );
    await expect(fetchReadiness()).rejects.toThrow("Request failed.");
  });

  it("reports timeout errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, init?: RequestInit) => {
        init?.signal?.dispatchEvent(new Event("abort"));
        return Promise.reject(new DOMException("aborted", "AbortError"));
      }),
    );
    await expect(fetchLiveness()).rejects.toMatchObject({ detail: "timeout" });
  });
});
