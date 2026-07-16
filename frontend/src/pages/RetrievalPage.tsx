import { useState } from "react";

import { EmptyState } from "../components/feedback/States";
import { PageHeader } from "../components/layout/PageHeader";
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { Section } from "../components/ui/Section";
import { Tabs } from "../components/ui/Tabs";

export function RetrievalPage() {
  const [mode, setMode] = useState("hybrid");
  return (
    <>
      <PageHeader
        title="Retrieval Playground"
        description="Prepare dense, sparse, and hybrid retrieval experiments with scoped metadata filters."
      />
      <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
        <Card className="space-y-5">
          <Input
            placeholder="Future query input"
            aria-label="Retrieval query"
            disabled
          />
          <Tabs
            value={mode}
            onChange={setMode}
            items={[
              { id: "dense", label: "Dense" },
              { id: "sparse", label: "Sparse" },
              { id: "hybrid", label: "Hybrid" },
            ]}
          />
          <div className="rounded-token border border-border bg-background p-4 text-sm text-muted">
            Filter controls will support document IDs and content types.
          </div>
        </Card>
        <Section
          title="Results and diagnostics"
          description="Result cards will show retrieved chunks and optional score diagnostics."
        >
          <EmptyState
            title="No retrieval request yet"
            description="Diagnostics will explain branch ranks, raw scores, and weighted-RRF contribution values."
          />
        </Section>
      </div>
    </>
  );
}
