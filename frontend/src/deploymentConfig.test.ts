import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

describe("deployment configuration", () => {
  it("uses same-origin API paths when VITE_API_BASE_URL is unset", () => {
    const client = readFileSync(join(process.cwd(), "src/lib/api/client.ts"), "utf-8");

    expect(client).toContain("import.meta.env.VITE_API_BASE_URL");
    expect(client).toContain('import.meta.env.VITE_API_BASE_URL ?? ""');
  });

  it("keeps local development and upload validation environment-driven without VITE secrets", () => {
    const utils = readFileSync(
      join(process.cwd(), "src/features/documents/utils.ts"),
      "utf-8",
    );
    const envExample = readFileSync(join(process.cwd(), ".env.example"), "utf-8");

    expect(utils).toContain("VITE_MAX_DOCUMENT_UPLOAD_BYTES");
    expect(utils).toContain("defaultMaxUploadBytes = 10 * 1024 * 1024");
    expect(envExample).toContain("VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1");
    expect(envExample).not.toContain("VITE_API_KEY");
    expect(envExample).not.toContain("VITE_QDRANT");
  });
});
