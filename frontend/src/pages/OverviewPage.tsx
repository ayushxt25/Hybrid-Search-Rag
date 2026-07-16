import {
  ArrowRight,
  Database,
  FileText,
  GitMerge,
  Layers,
  LifeBuoy,
  MessageSquareQuote,
  Search,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "../components/layout/PageHeader";
import { Card } from "../components/ui/Card";
import { Section } from "../components/ui/Section";
import { HealthStatus } from "../features/health/HealthStatus";

const capabilities: Array<{
  title: string;
  description: string;
  icon: LucideIcon;
}> = [
  {
    title: "Multi-format ingestion",
    description: "TXT, Markdown, PDF, and DOCX document entry points.",
    icon: FileText,
  },
  {
    title: "Dense retrieval",
    description: "Semantic search for meaning-oriented discovery.",
    icon: Search,
  },
  {
    title: "Sparse retrieval",
    description: "Deterministic lexical search over explicit terms.",
    icon: Layers,
  },
  {
    title: "Weighted-RRF hybrid search",
    description: "Dense and sparse rankings combined by weighted rank fusion.",
    icon: GitMerge,
  },
  {
    title: "Metadata-scoped retrieval",
    description: "Document and content-type filters applied before ranking.",
    icon: ShieldCheck,
  },
  {
    title: "Explainable ranking",
    description: "Optional branch ranks, scores, and RRF contribution diagnostics.",
    icon: LifeBuoy,
  },
  {
    title: "Grounded answers",
    description: "Answers assembled from retrieved sources with structured citations.",
    icon: MessageSquareQuote,
  },
  {
    title: "Document lifecycle management",
    description: "List, inspect, replace, and delete indexed documents.",
    icon: Database,
  },
];

export function OverviewPage() {
  return (
    <>
      <PageHeader
        title="Hybrid Search Studio"
        description="Explainable retrieval and grounded answers over internal documents."
      />
      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <div className="space-y-6">
          <Section
            title="Capabilities"
            description="Backend features surfaced as a focused developer console."
          >
            <div className="grid gap-4 md:grid-cols-2">
              {capabilities.map((capability) => (
                <Card key={capability.title} className="min-h-32">
                  <capability.icon className="mb-4 h-5 w-5 text-accent" />
                  <h3 className="font-semibold">{capability.title}</h3>
                  <p className="mt-2 text-sm leading-6 text-muted">
                    {capability.description}
                  </p>
                </Card>
              ))}
            </div>
          </Section>
          <Section title="Pipeline summary">
            <Card>
              <div className="flex flex-wrap items-center gap-2 text-sm text-secondary">
                {[
                  "Upload",
                  "Parse",
                  "Normalize",
                  "Chunk",
                  "Embed",
                  "Index",
                  "Retrieve",
                  "Ground",
                ].map((step, index, items) => (
                  <span key={step} className="flex items-center gap-2">
                    <span className="rounded border border-border bg-background px-3 py-1.5">
                      {step}
                    </span>
                    {index < items.length - 1 ? (
                      <ArrowRight className="h-4 w-4 text-muted" />
                    ) : null}
                  </span>
                ))}
              </div>
            </Card>
          </Section>
        </div>
        <aside className="space-y-4">
          <Card>
            <h2 className="font-semibold">System connection</h2>
            <p className="mt-2 text-sm text-muted">
              Liveness is checked through the backend health endpoint.
            </p>
            <div className="mt-4">
              <HealthStatus />
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold">Quick start</h2>
            <div className="mt-4 grid gap-2 text-sm">
              <Link
                className="rounded-token border border-border px-3 py-2 text-secondary hover:bg-elevated"
                to="/documents"
              >
                Prepare documents
              </Link>
              <Link
                className="rounded-token border border-border px-3 py-2 text-secondary hover:bg-elevated"
                to="/retrieval"
              >
                Explore retrieval
              </Link>
              <Link
                className="rounded-token border border-border px-3 py-2 text-secondary hover:bg-elevated"
                to="/answers"
              >
                Review grounded answers
              </Link>
            </div>
          </Card>
        </aside>
      </div>
    </>
  );
}
