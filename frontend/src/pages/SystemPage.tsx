import { PageHeader } from "../components/layout/PageHeader";
import { Card } from "../components/ui/Card";
import { CodeBlock } from "../components/ui/CodeBlock";
import { MetadataList } from "../components/ui/MetadataList";
import { Section } from "../components/ui/Section";
import { HealthStatus } from "../features/health/HealthStatus";

export function SystemPage() {
  return (
    <>
      <PageHeader
        title="System Health"
        description="Inspect safe backend connection and configuration placeholders."
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <h2 className="font-semibold">Liveness</h2>
          <div className="mt-4">
            <HealthStatus />
          </div>
        </Card>
        <Card>
          <h2 className="font-semibold">Readiness</h2>
          <p className="mt-2 text-sm text-muted">
            Readiness checks are not called automatically because local settings may
            disable them.
          </p>
        </Card>
      </div>
      <Section
        title="Backend endpoint"
        description="Public frontend configuration only."
      >
        <CodeBlock>
          {import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"}
        </CodeBlock>
      </Section>
      <Section title="Configuration placeholders">
        <MetadataList
          items={[
            { label: "Secrets", value: "Never rendered in the UI" },
            { label: "Diagnostics", value: "Safe metadata only" },
          ]}
        />
      </Section>
    </>
  );
}
