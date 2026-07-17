import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { CodeBlock } from "../components/ui/CodeBlock";
import { MetadataList } from "../components/ui/MetadataList";
import { Section } from "../components/ui/Section";
import {
  getApiBaseUrl,
  healthPaths,
  isSessionApiKeySet,
  type ComponentHealth,
} from "../features/health/api";
import { HealthStatusBadge } from "../features/health/HealthStatus";
import { useHealth, useReadinessCheck } from "../features/health/useHealth";
import { healthUiState } from "../features/health/state";

function timeLabel(value: Date | null) {
  return value ? value.toLocaleTimeString() : "Not checked yet";
}

function readinessState(
  readiness: ReturnType<typeof useReadinessCheck>,
): "not_checked" | "ready" | "not_ready" | "disabled" | "unavailable" {
  if (readiness.isIdle) return "not_checked";
  if (readiness.isError) return "unavailable";
  const response = readiness.data?.data;
  if (!response) return "not_checked";
  if (response.components.readiness?.status === "not_configured") return "disabled";
  return response.status === "ready" ? "ready" : "not_ready";
}

function readinessLabel(state: ReturnType<typeof readinessState>) {
  return {
    not_checked: "Not checked",
    ready: "Ready",
    not_ready: "Not ready",
    disabled: "Disabled",
    unavailable: "Unavailable",
  }[state];
}

function componentValue(component: ComponentHealth) {
  return component.detail
    ? `${component.status}: ${component.detail}`
    : component.status;
}

export function SystemPage() {
  const liveness = useHealth();
  const readiness = useReadinessCheck();
  const [lastLiveSuccess, setLastLiveSuccess] = useState<Date | null>(null);
  const [lastReadySuccess, setLastReadySuccess] = useState<Date | null>(null);
  const liveState = healthUiState(liveness);
  const readyState = readinessState(readiness);
  const apiBaseUrl = getApiBaseUrl();

  useEffect(() => {
    if (liveness.isSuccess) setLastLiveSuccess(new Date());
  }, [liveness.isSuccess]);

  async function refreshLiveness() {
    const result = await liveness.refetch();
    if (result.isSuccess) setLastLiveSuccess(new Date());
  }

  async function checkReadiness() {
    try {
      await readiness.mutateAsync();
      setLastReadySuccess(new Date());
    } catch {
      // Error state is rendered safely by the mutation result.
    }
  }

  return (
    <>
      <PageHeader
        title="System Health"
        description="Inspect safe operational status for the local Hybrid Search Studio stack."
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="font-semibold">Liveness</h2>
              <p className="mt-1 text-sm text-muted">Fast process health check.</p>
            </div>
            <HealthStatusBadge state={liveState} />
          </div>
          <MetadataList
            items={[
              {
                label: "Status",
                value:
                  liveState === "connected"
                    ? "Alive"
                    : liveState === "checking"
                      ? "Checking"
                      : "Unavailable",
              },
              { label: "Endpoint", value: healthPaths.live },
              { label: "Last successful check", value: timeLabel(lastLiveSuccess) },
              {
                label: "Request ID",
                value: liveness.data?.requestId ?? "Not returned",
              },
            ]}
          />
          <Button
            type="button"
            variant="secondary"
            className="mt-4"
            onClick={refreshLiveness}
            isLoading={liveness.isFetching}
          >
            <RefreshCw aria-hidden className="h-4 w-4" />
            Refresh liveness
          </Button>
        </Card>

        <Card>
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="font-semibold">Readiness</h2>
              <p className="mt-1 text-sm text-muted">Manual dependency check.</p>
            </div>
            <Badge
              tone={
                readyState === "ready"
                  ? "success"
                  : readyState === "not_ready" || readyState === "unavailable"
                    ? "danger"
                    : "neutral"
              }
            >
              {readinessLabel(readyState)}
            </Badge>
          </div>
          <MetadataList
            items={[
              { label: "Status", value: readinessLabel(readyState) },
              { label: "Endpoint", value: healthPaths.ready },
              { label: "Last successful check", value: timeLabel(lastReadySuccess) },
              {
                label: "Request ID",
                value: readiness.data?.requestId ?? "Not returned",
              },
            ]}
          />
          {readiness.data?.data.components ? (
            <div className="mt-4 space-y-2" aria-label="Readiness components">
              {Object.entries(readiness.data.data.components).map(
                ([name, component]) => (
                  <div
                    key={name}
                    className="flex flex-wrap items-center justify-between gap-3 rounded-token border border-border bg-background p-3 text-sm"
                  >
                    <span className="font-medium">{name}</span>
                    <span className="text-muted">{componentValue(component)}</span>
                  </div>
                ),
              )}
            </div>
          ) : null}
          {readiness.isError && (
            <p className="mt-4 text-sm text-danger" role="alert">
              Readiness is unavailable. Liveness and the rest of the UI can still be
              used.
            </p>
          )}
          <Button
            type="button"
            variant="secondary"
            className="mt-4"
            onClick={checkReadiness}
            isLoading={readiness.isPending}
            disabled={readiness.isPending}
          >
            Check readiness
          </Button>
        </Card>
      </div>

      <Section title="Connection" description="Public frontend configuration only.">
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <MetadataList
              items={[
                { label: "API base URL", value: apiBaseUrl || "Same origin /api" },
                {
                  label: "Backend connection",
                  value: liveState === "connected" ? "Connected" : liveState,
                },
                { label: "Browser origin", value: window.location.origin },
                {
                  label: "Session access key set",
                  value: isSessionApiKeySet() ? "Yes" : "No",
                },
              ]}
            />
          </Card>
          <CodeBlock>{apiBaseUrl || "/api"}</CodeBlock>
        </div>
      </Section>

      <Section title="Environment capabilities">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[
            "Supported files: TXT, Markdown, PDF, DOCX",
            "Retrieval: Dense, Sparse, Weighted RRF Hybrid",
            "Diagnostics: Available",
            "Document lifecycle: Upload, Detail, Replace, Delete",
            "Grounded generation: Configured by backend environment",
          ].map((capability) => (
            <Card key={capability} className="text-sm text-secondary">
              {capability}
            </Card>
          ))}
        </div>
      </Section>
    </>
  );
}
