import { afterEach, describe, expect, it, vi } from "vitest";

import { clearSessionApiKey, setSessionApiKey } from "../../lib/api/client";
import { askGroundedAnswer } from "./api";

function ok(data: unknown = { answer: "ok" }) {
  return Promise.resolve({
    ok: true,
    status: 200,
    headers: new Headers({
      "content-type": "application/json",
      "X-Request-ID": "req-answer",
    }),
    json: async () => data,
  } as Response);
}

describe("answer API", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uses the grounded-answer endpoint and exact payload", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    const result = await askGroundedAnswer({
      question: "What is covered?",
      limit: 4,
      candidate_limit: 12,
      document_ids: ["a".repeat(64)],
      content_types: ["application/pdf"],
    });
    expect(fetcher.mock.calls[0][0]).toBe(
      "http://127.0.0.1:8000/api/v1/answers/grounded",
    );
    expect(JSON.parse(fetcher.mock.calls[0][1].body as string)).toEqual({
      question: "What is covered?",
      limit: 4,
      candidate_limit: 12,
      document_ids: ["a".repeat(64)],
      content_types: ["application/pdf"],
    });
    expect(result.requestId).toBe("req-answer");
  });

  it("uses the in-memory API key header", async () => {
    const fetcher = vi.fn().mockResolvedValue(await ok());
    vi.stubGlobal("fetch", fetcher);
    setSessionApiKey("answer-key");
    await askGroundedAnswer({ question: "q", limit: 5, candidate_limit: 20 });
    expect((fetcher.mock.calls[0][1].headers as Headers).get("X-API-Key")).toBe(
      "answer-key",
    );
  });

  it("preserves Retry-After and handles non-JSON errors safely", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 429,
        headers: new Headers({ "Retry-After": "15", "content-type": "text/plain" }),
        json: async () => ({}),
      } as Response),
    );
    await expect(
      askGroundedAnswer({ question: "q", limit: 5, candidate_limit: 20 }),
    ).rejects.toMatchObject({
      message: "Request failed.",
      retryAfter: "15",
    });
  });
});
