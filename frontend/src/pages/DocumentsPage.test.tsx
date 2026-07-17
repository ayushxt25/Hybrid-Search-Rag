import "@testing-library/jest-dom/vitest";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createQueryClient } from "../app/queryClient";
import { clearSessionApiKey } from "../lib/api/client";
import { DocumentsPage } from "./DocumentsPage";
import { documentKeys } from "../features/documents/hooks";

const documentId = "a".repeat(64);
const replacementDocumentId = "c".repeat(64);
const detail = {
  document_id: documentId,
  filename: "policy.md",
  content_type: "text/markdown",
  content_hash: "b".repeat(64),
  chunk_count: 3,
  indexed_at: "2026-07-17T00:00:00Z",
  chunk_indices: [0, 1, 2],
  page_numbers: [],
  headings: ["Overview"],
};

function renderPage() {
  const client = createQueryClient();
  const view = render(
    <QueryClientProvider client={client}>
      <DocumentsPage />
    </QueryClientProvider>,
  );
  return { client, ...view };
}

function response(data: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => data,
  } as Response);
}

function mockFetch(
  documents = [detail],
  replacement = { documentId, filename: "policy.md" },
) {
  return vi.fn((url: string, init?: RequestInit) => {
    if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
    if (url.endsWith("/api/v1/documents") && init?.method !== "DELETE") {
      return response({ documents, next_cursor: null });
    }
    if (url.endsWith(`/api/v1/documents/${documentId}`) && init?.method === "DELETE") {
      return response({ document_id: documentId, deleted_chunks: 3, deleted: true });
    }
    if (url.endsWith(`/api/v1/documents/${documentId}`)) return response(detail);
    if (url.endsWith("/api/v1/documents/ingest")) {
      return response({
        document_id: replacement.documentId,
        content_hash: "b".repeat(64),
        file_name: replacement.filename,
        file_extension: ".md",
        chunk_count: 3,
        indexed_points: 3,
      });
    }
    return response({ detail: "not found" }, 404);
  });
}

describe("DocumentsPage", () => {
  afterEach(() => {
    clearSessionApiKey();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders loading then an empty state", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    renderPage();
    expect(screen.getByText(/Upload document/i)).toBeInTheDocument();
    expect(await screen.findByText("No indexed documents")).toBeInTheDocument();
  });

  it("renders document metadata and mobile-safe cards", async () => {
    vi.stubGlobal("fetch", mockFetch());
    renderPage();
    expect(
      await screen.findByText("Document operations are available."),
    ).toBeInTheDocument();
    expect(await screen.findAllByText("policy.md")).not.toHaveLength(0);
    expect(screen.getAllByText("text/markdown").length).toBeGreaterThan(0);
    expect(screen.getByText("1 returned")).toBeInTheDocument();
  });

  it("shows checking backend copy while liveness is pending", async () => {
    const fetcher = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/v1/health/live")) return new Promise(() => undefined);
      return mockFetch([])(url, init);
    });
    vi.stubGlobal("fetch", fetcher);
    renderPage();
    expect(screen.getByText("Checking backend availability.")).toBeInTheDocument();
    expect(screen.getByText("Checking")).toBeInTheDocument();
  });

  it("rejects unsupported uploads", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup({ applyAccept: false });
    renderPage();
    const input = screen.getByLabelText("Choose document file");
    await user.upload(
      input,
      new File(["x"], "bad.exe", { type: "application/x-msdownload" }),
    );
    expect(
      await screen.findByText("Use a TXT, Markdown, PDF, or DOCX file."),
    ).toBeInTheDocument();
  });

  it("rejects oversized uploads", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();
    const input = screen.getByLabelText("Choose document file");
    const file = new File([new Uint8Array(10 * 1024 * 1024 + 1)], "huge.pdf", {
      type: "application/pdf",
    });
    await user.upload(input, file);
    expect(await screen.findByText("File is larger than 10 MB.")).toBeInTheDocument();
  });

  it("uploads and invalidates the list", async () => {
    const fetcher = mockFetch();
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.upload(
      screen.getByLabelText("Choose document file"),
      new File(["x"], "ok.txt", { type: "text/plain" }),
    );
    await user.click(screen.getByRole("button", { name: "Upload" }));
    expect(await screen.findByText("Document uploaded.")).toBeInTheDocument();
    expect(
      fetcher.mock.calls.some(([url]) =>
        String(url).endsWith("/api/v1/documents/ingest"),
      ),
    ).toBe(true);
  });

  it("shows safe upload API errors", async () => {
    const fetcher = mockFetch();
    fetcher.mockImplementation((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents"))
        return response({ documents: [], next_cursor: null });
      return response({ detail: "Valid API credentials are required." }, 401);
    });
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup({ applyAccept: false });
    renderPage();
    await user.upload(
      screen.getByLabelText("Choose document file"),
      new File(["x"], "ok.txt", { type: "text/plain" }),
    );
    await user.click(screen.getByRole("button", { name: "Upload" }));
    expect(
      (await screen.findAllByText(/API credentials were not accepted/i)).length,
    ).toBeGreaterThan(0);
  });

  it("opens detail and copies the full ID", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn() },
    });
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();
    await user.click((await screen.findAllByRole("button", { name: "Inspect" }))[0]);
    expect(await screen.findByText(documentId)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy document ID" }));
    expect(await screen.findByText("Document ID copied.")).toBeInTheDocument();
  });

  it("requires replacement confirmation and sends selected document ID", async () => {
    const fetcher = mockFetch([detail], {
      documentId: replacementDocumentId,
      filename: "replacement.md",
    });
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    const { client } = renderPage();
    await user.click((await screen.findAllByRole("button", { name: "Inspect" }))[0]);
    await screen.findByText(documentId);
    expect(client.getQueryData(documentKeys.detail(documentId))).toBeDefined();
    const detailGetsBeforeReplace = fetcher.mock.calls.filter(([url, init]) => {
      return (
        String(url).endsWith(`/api/v1/documents/${documentId}`) &&
        init?.method !== "DELETE"
      );
    }).length;
    await user.upload(
      await screen.findByLabelText("Replacement file"),
      new File(["x"], "new.md", { type: "text/markdown" }),
    );
    await user.click(screen.getByRole("button", { name: /Replace/i }));
    await user.click(
      within(screen.getByRole("dialog", { name: "Replace document?" })).getByRole(
        "button",
        { name: "Replace" },
      ),
    );
    await waitFor(() => {
      const ingest = fetcher.mock.calls.find(([url]) =>
        String(url).endsWith("/api/v1/documents/ingest"),
      );
      expect(ingest).toBeDefined();
      expect((ingest?.[1]?.body as FormData).get("replace_document_id")).toBe(
        documentId,
      );
    });
    expect(await screen.findByText("replacement.md replaced.")).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "policy.md" })).not.toBeInTheDocument();
    expect(client.getQueryData(documentKeys.detail(documentId))).toBeUndefined();
    const detailGetsAfterReplace = fetcher.mock.calls.filter(([url, init]) => {
      return (
        String(url).endsWith(`/api/v1/documents/${documentId}`) &&
        init?.method !== "DELETE"
      );
    }).length;
    expect(detailGetsAfterReplace).toBe(detailGetsBeforeReplace);
    expect(
      fetcher.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/documents/ingest"),
      ),
    ).toHaveLength(1);
  });

  it("supports same-ID replacement and closes detail", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();
    await user.click((await screen.findAllByRole("button", { name: "Inspect" }))[0]);
    await user.upload(
      await screen.findByLabelText("Replacement file"),
      new File(["x"], "same.md", { type: "text/markdown" }),
    );
    await user.click(screen.getByRole("button", { name: /Replace/i }));
    await user.click(
      within(screen.getByRole("dialog", { name: "Replace document?" })).getByRole(
        "button",
        { name: "Replace" },
      ),
    );
    expect(await screen.findByText("policy.md replaced.")).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "policy.md" })).not.toBeInTheDocument();
  });

  it("shows unavailable state for a genuine detail 404", async () => {
    const fetcher = mockFetch();
    fetcher.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/v1/health/live")) return response({ status: "alive" });
      if (url.endsWith("/api/v1/documents") && init?.method !== "DELETE") {
        return response({ documents: [detail], next_cursor: null });
      }
      if (url.endsWith(`/api/v1/documents/${documentId}`)) {
        return response({ detail: "Indexed document was not found." }, 404);
      }
      return response({ detail: "not found" }, 404);
    });
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.click((await screen.findAllByRole("button", { name: "Inspect" }))[0]);
    expect(await screen.findByText("Document unavailable")).toBeInTheDocument();
  });

  it("requires delete confirmation and clears stale UI", async () => {
    const fetcher = mockFetch();
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await user.click(
      (await screen.findAllByRole("button", { name: /Delete policy.md/i }))[0],
    );
    await user.click(
      within(screen.getByRole("dialog", { name: "Delete document?" })).getByRole(
        "button",
        { name: "Delete" },
      ),
    );
    expect(await screen.findByText("Document deleted.")).toBeInTheDocument();
  });

  it("disables write actions when backend is unavailable", async () => {
    const fetcher = vi.fn((url: string) => {
      if (url.endsWith("/api/v1/health/live")) return response({ detail: "down" }, 503);
      return response({ documents: [], next_cursor: null });
    });
    vi.stubGlobal("fetch", fetcher);
    renderPage();
    await screen.findByText("Unavailable");
    expect(
      screen.getByText("Writes are disabled while the backend is unavailable."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload" })).toBeDisabled();
  });

  it("does not render API keys", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    const user = userEvent.setup();
    renderPage();
    await user.type(screen.getByLabelText("API key"), "sk-secret");
    await user.click(screen.getByRole("button", { name: "Set" }));
    expect(screen.queryByDisplayValue("sk-secret")).not.toBeInTheDocument();
    expect(screen.queryByText("sk-secret")).not.toBeInTheDocument();
  });
});
