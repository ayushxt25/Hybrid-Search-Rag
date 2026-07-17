import "@testing-library/jest-dom/vitest";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createQueryClient } from "../app/queryClient";
import { clearSessionApiKey } from "../lib/api/client";
import { RetrievalPage } from "./RetrievalPage";

const docId = "a".repeat(64);
const otherDocId = "b".repeat(64);

const documents = [
  {
    document_id: docId,
    filename: "policy.md",
    content_type: "text/markdown",
    content_hash: "h",
    chunk_count: 2,
    indexed_at: null,
  },
  {
    document_id: otherDocId,
    filename: "manual.pdf",
    content_type: "application/pdf",
    content_hash: "i",
    chunk_count: 1,
    indexed_at: null,
  },
];

function renderPage() {
  return render(
    <QueryClientProvider client={createQueryClient()}>
      <RetrievalPage />
    </QueryClientProvider>,
  );
}

function response(data: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => data,
  } as Response);
}

function result(text = "retrieved marker text") {
  return {
    query: "marker",
    result_count: 1,
    results: [
      {
        point_id: "hidden-point-id",
        chunk_id: "c".repeat(64),
        document_id: docId,
        score: 0.42,
        file_name: "policy.md",
        file_extension: ".md",
        chunk_index: 0,
        section_index: 0,
        page_number: null,
        heading: "Scope",
        text,
        start_word: 0,
        end_word: 3,
        word_count: 3,
        score_diagnostics: {
          dense: {
            raw_score: 0.9,
            rank: 1,
            weight: 1.5,
            rrf_contribution: 0.02459,
          },
          sparse: {
            raw_score: null,
            rank: null,
            weight: 1,
            rrf_contribution: 0,
          },
          fused_score: 0.02459,
          fused_rank: 1,
        },
      },
    ],
  };
}

function mockFetch(searchData = result()) {
  return vi.fn((url: string, init?: RequestInit) => {
    void init;
    if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
    if (url.endsWith("/api/v1/documents")) {
      return response({ documents, next_cursor: null });
    }
    if (url.includes("/api/v1/search/")) {
      return response(searchData);
    }
    return response({ detail: "not found" }, 404);
  });
}

describe("RetrievalPage", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("rejects blank query and disables search", async () => {
    vi.stubGlobal("fetch", mockFetch());
    renderPage();
    expect(await screen.findByText("Enter a retrieval query.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Search/i })).toBeDisabled();
  });

  it("maps document and content-type filters to the search payload", async () => {
    const fetcher = mockFetch();
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText("Query"), "marker");
    await user.click(await screen.findByText("policy.md"));
    await user.click(screen.getByText("Markdown"));
    await user.click(screen.getByRole("button", { name: /Search/i }));

    await screen.findByText("retrieved marker text");
    const searchCall = fetcher.mock.calls.find(([url]) =>
      String(url).endsWith("/api/v1/search/hybrid"),
    );
    const body = JSON.parse(searchCall?.[1]?.body as string);
    expect(body.document_ids).toEqual([docId]);
    expect(body.content_types).toEqual(["text/markdown"]);
    expect(body.candidate_limit).toBe(20);
  });

  it("switches modes and validates candidate limit", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Query"), "marker");
    await user.click(screen.getByRole("tab", { name: "Dense" }));
    expect(screen.queryByLabelText(/Candidate limit/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Hybrid" }));
    await user.clear(screen.getByLabelText(/Candidate limit/i));
    await user.type(screen.getByLabelText(/Candidate limit/i), "1");
    expect(
      screen.getByText(
        "Candidate limit must be greater than or equal to result limit.",
      ),
    ).toBeInTheDocument();
  });

  it("renders hybrid results and diagnostics without internals", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Query"), "marker");
    await user.click(screen.getByLabelText(/Include score diagnostics/i));
    await user.click(screen.getByRole("button", { name: /Search/i }));

    expect(await screen.findByText("retrieved marker text")).toBeInTheDocument();
    expect(screen.getByText("Score diagnostics")).toBeInTheDocument();
    expect(screen.getAllByText("Not present in branch")).toHaveLength(2);
    expect(screen.getByText(/Fused score is not a probability/i)).toBeInTheDocument();
    expect(screen.queryByText("hidden-point-id")).not.toBeInTheDocument();
    expect(screen.queryByText(/vector/i)).not.toBeInTheDocument();
  });

  it("expands long text and copies chunk text", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn() },
    });
    vi.stubGlobal("fetch", mockFetch(result("long ".repeat(150))));
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Query"), "marker");
    await user.click(screen.getByRole("button", { name: /Search/i }));
    expect(
      await screen.findByRole("button", { name: "Show more" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy chunk text" }));
    expect(await screen.findByText("Copied.")).toBeInTheDocument();
  });

  it("shows empty and validation error states safely", async () => {
    const fetcher = mockFetch({ query: "none", result_count: 0, results: [] });
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("Query"), "none");
    await user.click(screen.getByRole("button", { name: /Search/i }));
    expect(await screen.findByText("No matching chunks")).toBeInTheDocument();

    fetcher.mockImplementation((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents")) {
        return response({ documents, next_cursor: null });
      }
      return response({ detail: [{ msg: "server detail" }] }, 422);
    });
    await user.clear(screen.getByLabelText("Query"));
    await user.type(screen.getByLabelText("Query"), "bad");
    await user.click(screen.getByRole("button", { name: /Search/i }));
    expect(
      await screen.findByText("Check the query, limits, and filters."),
    ).toBeInTheDocument();
  });

  it("disables search when backend is unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.endsWith("/api/v1/health/live")) {
          return response({ detail: "down" }, 503);
        }
        return response({ documents, next_cursor: null });
      }),
    );
    renderPage();
    await screen.findByText("Backend is unavailable.");
    expect(screen.getByRole("button", { name: /Search/i })).toBeDisabled();
  });

  it("prevents duplicate in-flight submissions", async () => {
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
    await user.type(screen.getByLabelText("Query"), "marker");
    await user.click(screen.getByRole("button", { name: /Search/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Loading..." })).toBeDisabled(),
    );
    expect(
      fetcher.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/search/hybrid"),
      ),
    ).toHaveLength(1);
  });
});
