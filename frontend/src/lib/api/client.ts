export class ApiError extends Error {
  status: number;
  requestId?: string;
  retryAfter?: string;
  detail: string;

  constructor(
    message: string,
    options: {
      status: number;
      requestId?: string;
      retryAfter?: string;
      detail: string;
    },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.requestId = options.requestId;
    this.retryAfter = options.retryAfter;
    this.detail = options.detail;
  }
}

export type ApiClientOptions = {
  baseUrl?: string;
  timeoutMs?: number;
  getApiKey?: () => string | undefined;
  fetcher?: typeof fetch;
};

export type ApiResult<T> = {
  data: T;
  requestId?: string;
};

const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
let sessionApiKey: string | undefined;

export function setSessionApiKey(value: string | undefined) {
  const normalized = value?.trim();
  sessionApiKey = normalized || undefined;
}

export function clearSessionApiKey() {
  sessionApiKey = undefined;
}

export function hasSessionApiKey() {
  return Boolean(sessionApiKey);
}

async function readBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return undefined;
  }
  return response.json();
}

function detailFrom(body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    return String((body as { detail: unknown }).detail);
  }
  return "Request failed.";
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly getApiKey?: () => string | undefined;
  private readonly fetcher?: typeof fetch;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs ?? 10000;
    this.getApiKey = options.getApiKey;
    this.fetcher = options.fetcher;
  }

  async json<T>(path: string, init: RequestInit = {}): Promise<ApiResult<T>> {
    return this.request<T>(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  }

  async multipart<T>(path: string, formData: FormData): Promise<ApiResult<T>> {
    return this.request<T>(path, { method: "POST", body: formData });
  }

  private async request<T>(path: string, init: RequestInit): Promise<ApiResult<T>> {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), this.timeoutMs);
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    const apiKey = this.getApiKey?.();
    if (apiKey) {
      headers.set("X-API-Key", apiKey);
    }

    try {
      const response = await (this.fetcher ?? fetch)(this.baseUrl + path, {
        ...init,
        headers,
        signal: controller.signal,
      });
      const requestId = response.headers.get("X-Request-ID") ?? undefined;
      const retryAfter = response.headers.get("Retry-After") ?? undefined;
      const body = await readBody(response);
      if (!response.ok) {
        const detail = detailFrom(body);
        throw new ApiError(detail, {
          status: response.status,
          requestId,
          retryAfter,
          detail,
        });
      }
      return { data: body as T, requestId };
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("Request timed out.", { status: 0, detail: "timeout" });
      }
      throw new ApiError("Backend request failed.", {
        status: 0,
        detail: "network_error",
      });
    } finally {
      window.clearTimeout(timeout);
    }
  }
}

export const apiClient = new ApiClient({ getApiKey: () => sessionApiKey });
