import "@testing-library/jest-dom/vitest";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createQueryClient } from "../app/queryClient";
import { clearSessionApiKey, setSessionApiKey } from "../lib/api/client";
import { SystemPage } from "./SystemPage";

function renderPage() {
  return render(
    <QueryClientProvider client={createQueryClient()}>
      <SystemPage />
    </QueryClientProvider>,
  );
}

function response(data: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({
      "content-type": "application/json",
      "X-Request-ID": "rid-system",
    }),
    json: async () => data,
  } as Response);
}

function mockFetch({ live = true, readyStatus = "ready" } = {}) {
  return vi.fn((url: string) => {
    if (url.endsWith("/api/v1/health/live")) {
      return live ? response({ status: "alive" }) : response({ detail: "down" }, 503);
    }
    if (url.endsWith("/api/v1/health/ready")) {
      if (readyStatus === "error") return response({ detail: "down" }, 503);
      if (readyStatus === "disabled") {
        return response({
          status: "ready",
          components: {
            readiness: {
              status: "not_configured",
              detail: "Readiness checks are disabled.",
            },
          },
        });
      }
      return response(
        {
          status: readyStatus,
          components: {
            qdrant: {
              status: readyStatus === "ready" ? "healthy" : "unhealthy",
              detail:
                readyStatus === "ready" ? null : "Vector database is unavailable.",
            },
            generation: { status: "healthy" },
          },
        },
        readyStatus === "ready" ? 200 : 503,
      );
    }
    return response({ detail: "not found" }, 404);
  });
}

describe("SystemPage", () => {
  afterEach(() => {
    act(() => clearSessionApiKey());
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders liveness and safe connection information", async () => {
    vi.stubGlobal("fetch", mockFetch());
    renderPage();
    expect(await screen.findAllByText("Connected")).not.toHaveLength(0);
    expect(screen.getByText("/api/v1/health/live")).toBeInTheDocument();
    expect(screen.getAllByText("/api/v1").length).toBeGreaterThan(0);
    expect(screen.getByText(window.location.origin)).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("shows unavailable liveness", async () => {
    vi.stubGlobal("fetch", mockFetch({ live: false }));
    renderPage();
    expect(await screen.findAllByText("Unavailable")).not.toHaveLength(0);
  });

  it("does not call readiness on initial render and calls it after explicit action", async () => {
    const fetcher = mockFetch();
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await screen.findAllByText("Connected");
    expect(
      fetcher.mock.calls.some(([url]) => String(url).endsWith("/api/v1/health/ready")),
    ).toBe(false);
    await user.click(screen.getByRole("button", { name: "Check readiness" }));
    expect(await screen.findByText("qdrant")).toBeInTheDocument();
    expect(
      fetcher.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/health/ready"),
      ),
    ).toHaveLength(1);
  });

  it("renders ready, not-ready, disabled, and unavailable readiness states", async () => {
    const user = userEvent.setup();
    const fetcher = mockFetch({ readyStatus: "not_ready" });
    vi.stubGlobal("fetch", fetcher);
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Check readiness" }));
    expect(await screen.findAllByText("Not ready")).not.toHaveLength(0);

    fetcher.mockImplementation(mockFetch({ readyStatus: "disabled" }));
    await user.click(screen.getByRole("button", { name: "Check readiness" }));
    expect(await screen.findAllByText("Disabled")).not.toHaveLength(0);

    fetcher.mockImplementation(mockFetch({ readyStatus: "error" }));
    await user.click(screen.getByRole("button", { name: "Check readiness" }));
    expect(await screen.findByText(/Readiness is unavailable/i)).toBeInTheDocument();
  });

  it("manual liveness refresh creates one request", async () => {
    const fetcher = mockFetch();
    vi.stubGlobal("fetch", fetcher);
    const user = userEvent.setup();
    renderPage();
    await screen.findAllByText("Connected");
    const before = fetcher.mock.calls.filter(([url]) =>
      String(url).endsWith("/api/v1/health/live"),
    ).length;
    await user.click(screen.getByRole("button", { name: /Refresh liveness/i }));
    await waitFor(() => {
      const after = fetcher.mock.calls.filter(([url]) =>
        String(url).endsWith("/api/v1/health/live"),
      ).length;
      expect(after).toBe(before + 1);
    });
  });

  it("shows API key presence only as yes/no", async () => {
    act(() => setSessionApiKey("sk-secret-system"));
    vi.stubGlobal("fetch", mockFetch());
    renderPage();
    expect(await screen.findByText("Yes")).toBeInTheDocument();
    expect(screen.queryByText("sk-secret-system")).not.toBeInTheDocument();
  });

  it("saves, updates, and clears the session API key without displaying it", async () => {
    vi.stubGlobal("fetch", mockFetch());
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByText("No")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Session API key"), "  sk-secret-system  ");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Yes")).toBeInTheDocument();
    expect(screen.getByLabelText("Session API key")).toHaveValue("");
    expect(screen.queryByDisplayValue("sk-secret-system")).not.toBeInTheDocument();
    expect(screen.queryByText("sk-secret-system")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Update" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(await screen.findByText("No")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument();
  });

  it("renders factual capabilities", async () => {
    vi.stubGlobal("fetch", mockFetch());
    renderPage();
    expect(
      await screen.findByText("Supported files: TXT, Markdown, PDF, DOCX"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Retrieval: Dense, Sparse, Weighted RRF Hybrid"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Grounded generation: Configured by backend environment"),
    ).toBeInTheDocument();
  });
});
