import "@testing-library/jest-dom/vitest";

import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderApp } from "../test/render";

function mockHealth(ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok,
      status: ok ? 200 : 503,
      headers: new Headers({
        "content-type": "application/json",
        "X-Request-ID": "rid-1",
      }),
      json: async () => (ok ? { status: "alive" } : { detail: "down" }),
    }),
  );
}

describe("app routing and shell", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  beforeEach(() => {
    mockHealth(true);
  });

  it("redirects root to overview", async () => {
    renderApp("/");
    expect(
      await screen.findByRole("heading", { name: "Hybrid Search Studio" }),
    ).toBeInTheDocument();
  });

  it("renders sidebar navigation and marks the current route", async () => {
    renderApp("/retrieval");
    expect(
      await screen.findByRole("heading", { name: "Retrieval Playground" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Retrieval Playground/i })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("opens and closes mobile navigation", async () => {
    const user = userEvent.setup();
    renderApp("/overview");
    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(
      screen.getByRole("dialog", { name: "Mobile navigation" }),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Mobile navigation" }),
      ).not.toBeInTheDocument(),
    );
  });

  it("renders overview capability cards", async () => {
    renderApp("/overview");
    expect(await screen.findByText("Multi-format ingestion")).toBeInTheDocument();
    expect(screen.getByText("Weighted-RRF hybrid search")).toBeInTheDocument();
    expect(screen.getByText("Explainable ranking")).toBeInTheDocument();
  });

  it("shows connected health status for a successful liveness response", async () => {
    renderApp("/overview");
    expect(await screen.findAllByText("Connected")).not.toHaveLength(0);
  });

  it("shows unavailable health status for a failed liveness response", async () => {
    mockHealth(false);
    renderApp("/overview");
    expect(await screen.findAllByText("Unavailable")).not.toHaveLength(0);
  });

  it("renders a polished not-found route", async () => {
    renderApp("/missing");
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
  });

  it("does not render API keys or secrets", async () => {
    renderApp("/system");
    expect(
      await screen.findByRole("heading", { name: "System Health" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/sk-/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/api key/i)).not.toBeInTheDocument();
  });
});
