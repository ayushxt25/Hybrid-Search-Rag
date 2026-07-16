import { EmptyState } from "../components/feedback/States";
import { PageHeader } from "../components/layout/PageHeader";
import { Card } from "../components/ui/Card";
import { Textarea } from "../components/ui/Textarea";

export function AnswersPage() {
  return (
    <>
      <PageHeader
        title="Grounded Answers"
        description="Compose grounded questions and inspect cited evidence sources."
      />
      <div className="grid gap-6 lg:grid-cols-[420px_1fr]">
        <Card className="space-y-4">
          <Textarea
            placeholder="Future grounded question input"
            aria-label="Grounded question"
            disabled
          />
          <p className="text-sm text-muted">
            Answers will require retrieved context and validated citation markers.
          </p>
        </Card>
        <div className="space-y-4">
          <EmptyState
            title="Answer panel placeholder"
            description="Generated answers will appear here with citation markers."
          />
          <Card>
            <h2 className="font-semibold">Source cards</h2>
            <p className="mt-2 text-sm text-muted">
              Future source cards will expose safe chunk metadata and citation context.
            </p>
          </Card>
        </div>
      </div>
    </>
  );
}
