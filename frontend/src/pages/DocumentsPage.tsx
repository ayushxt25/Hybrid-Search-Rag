import { Upload } from "lucide-react";

import { EmptyState } from "../components/feedback/States";
import { PageHeader } from "../components/layout/PageHeader";
import { Card } from "../components/ui/Card";
import { Section } from "../components/ui/Section";

export function DocumentsPage() {
  return (
    <>
      <PageHeader
        title="Documents"
        description="Manage ingestion and lifecycle workflows for internal source documents."
      />
      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <Card className="border-dashed">
          <Upload className="mb-4 h-6 w-6 text-accent" />
          <h2 className="font-semibold">Upload area</h2>
          <p className="mt-2 text-sm text-muted">
            Future document upload controls will support TXT, Markdown, PDF, and DOCX
            files.
          </p>
        </Card>
        <Section title="Indexed documents">
          <EmptyState
            title="No document table yet"
            description="The foundation reserves space for listing, detail, replacement, and deletion workflows."
          />
        </Section>
      </div>
    </>
  );
}
