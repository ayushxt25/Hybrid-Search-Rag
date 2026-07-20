import "@testing-library/jest-dom/vitest";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createQueryClient } from "../app/queryClient";
import { clearSessionApiKey } from "../lib/api/client";
import { AnswersPage } from "./AnswersPage";

const docId = "a".repeat(64);
const chunkId = "c".repeat(64);
const documents = [
  {
    document_id: docId,
    filename: "handbook.pdf",
    content_type: "application/pdf",
    content_hash: "h",
    chunk_count: 2,
    indexed_at: null,
  },
];

function renderPage() {
  return render(
    <QueryClientProvider client={createQueryClient()}>
      <AnswersPage />
    </QueryClientProvider>,
  );
}

function response(data: unknown, status = 200, headers: Record<string, string> = {}) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "application/json", ...headers }),
    json: async () => data,
  } as Response);
}

function answerData(answer = "Policy applies to internal docs. [Source 1]") {
  return {
    question: "What applies?",
    answer,
    model_name: "deterministic-acceptance-provider",
    citations: [
      {
        source_number: 1,
        chunk_id: chunkId,
        document_id: docId,
        file_name: "handbook.pdf",
        heading: "Policy section",
        page_number: 2,
      },
    ],
    citation_markers: [1],
    retrieved_result_count: 1,
    context_source_count: 1,
    context_truncated: false,
    insufficient_context: false,
    input_characters: 10,
    output_characters: answer.length,
    finish_reason: "stop",
  };
}

function mockFetch(answer = answerData()) {
  return vi.fn((url: string, init?: RequestInit) => {
    void init;
    if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
    if (url.endsWith("/api/v1/documents")) {
      return response({ documents, next_cursor: null });
    }
    if (url.endsWith("/api/v1/answers/grounded")) {
      return response(answer, 200, { "X-Request-ID": "req-123" });
    }
    return response({ detail: "not found" }, 404);
  });
}

describe("AnswersPage", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("rejects blank questions and disables ask", async () => {
    vi.stubGlobal("fetch", mockFetch());
    renderPage();
    expect(await screen.findByText("Enter a grounded question.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ask/i })).toBeDisabled();
  });

  it("maps filters and submits exact answer payload", async () => {
    const fetcher = mockFetch();
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "What applies?");
    await user.click(await screen.findByText("handbook.pdf"));
    await user.click(screen.getByText("PDF"));
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect(await screen.findByText(/Policy applies/)).toBeInTheDocument();
    const call = fetcher.mock.calls.find(([url]) =>
      String(url).endsWith("/api/v1/answers/grounded"),
    );
    expect(JSON.parse(call?.[1]?.body as string)).toMatchObject({
      question: "What applies?",
      limit: 5,
      candidate_limit: 20,
      document_ids: [docId],
      content_types: ["application/pdf"],
    });
  });

  it("preserves paragraphs, citation markers, counts, request ID, and duration", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(answerData("First paragraph. [Source 1]\n\nSecond paragraph.")),
    );
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "What applies?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect(await screen.findByText("First paragraph. [Source 1]")).toBeInTheDocument();
    expect(screen.getByText("Second paragraph.")).toBeInTheDocument();
    expect(screen.getByText("1 citations")).toBeInTheDocument();
    expect(screen.getByText("1 sources")).toBeInTheDocument();
    expect(screen.getByText("Request req-123")).toBeInTheDocument();
    expect(screen.getByText(/client-observed duration/)).toBeInTheDocument();
    expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument();
  });

  it("clicks citation markers and highlights source", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const scrollIntoView = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollIntoView;
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "What applies?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    await user.click(await screen.findByRole("button", { name: "[Source 1]" }));
    expect(scrollIntoView).toHaveBeenCalled();
    expect(screen.getByText("Selected source")).toBeInTheDocument();
  });

  it("renders safe source metadata and copy actions without internals", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn() },
    });
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "What applies?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect((await screen.findAllByText("handbook.pdf")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("application/pdf").length).toBeGreaterThan(0);
    expect(screen.getByText("Policy section")).toBeInTheDocument();
    expect(screen.queryByText(chunkId)).not.toBeInTheDocument();
    expect(screen.queryByText(/prompt/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy source text" }));
    expect(await screen.findByText("Copied.")).toBeInTheDocument();
  });

  it("shows insufficient-context state without fabricating an answer", async () => {
    const insufficient = {
      ...answerData(
        "The provided documents do not contain enough information to answer this question.",
      ),
      citations: [],
      citation_markers: [],
      retrieved_result_count: 0,
      context_source_count: 0,
      insufficient_context: true,
      finish_reason: "insufficient_context",
      output_characters:
        "The provided documents do not contain enough information to answer this question."
          .length,
    };
    vi.stubGlobal("fetch", mockFetch(insufficient));
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "Unknown?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect(await screen.findByText("Not enough evidence")).toBeInTheDocument();
    expect(
      screen.getByText(/declined to generate an unsupported answer/i),
    ).toBeInTheDocument();
    expect(screen.getByText("0 sources")).toBeInTheDocument();
  });

  it("handles 429 Retry-After and provider errors safely", async () => {
    const fetcher = mockFetch();
    fetcher.mockImplementation((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents")) {
        return response({ documents, next_cursor: null });
      }
      return response({ detail: "limit" }, 429, { "Retry-After": "30" });
    });
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "What applies?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect(await screen.findByText(/Retry after 30 seconds/)).toBeInTheDocument();

    fetcher.mockImplementation((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents")) {
        return response({ documents, next_cursor: null });
      }
      return response({ detail: "provider body" }, 502);
    });
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect(
      await screen.findByText("The generation provider returned an invalid response."),
    ).toBeInTheDocument();

    fetcher.mockImplementation((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents")) {
        return response({ documents, next_cursor: null });
      }
      return response({ detail: "Valid API credentials are required." }, 401);
    });
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    expect(
      await screen.findByText(
        /Go to System Health -> Session API key, update the key, and retry/i,
      ),
    ).toBeInTheDocument();
  });

  it("disables ask when backend is unavailable and reset clears answer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.endsWith("/api/v1/health/live"))
          return response({ detail: "down" }, 503);
        return response({ documents, next_cursor: null });
      }),
    );
    renderPage();
    await screen.findByText("Backend is unavailable.");
    expect(screen.getByRole("button", { name: /Ask/i })).toBeDisabled();
  });

  it("prevents duplicate submissions while pending", async () => {
    const fetcher = vi.fn((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents")) {
        return response({ documents, next_cursor: null });
      }
      return new Promise<Response>(() => undefined);
    });
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Question"), "What applies?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Loading..." })).toBeDisabled(),
    );
    expect(
      fetcher.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/answers/grounded"),
      ),
    ).toHaveLength(1);
  });
});
