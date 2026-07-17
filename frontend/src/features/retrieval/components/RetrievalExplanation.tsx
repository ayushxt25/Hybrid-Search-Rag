import { Section } from "../../../components/ui/Section";

export function RetrievalExplanation() {
  return (
    <Section
      title="Ranking notes"
      description="Dense, sparse, and hybrid scores have different meanings."
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-token border border-border bg-surface p-4 text-sm text-muted">
          Dense retrieval uses semantic similarity for meaning-oriented matches. Sparse
          retrieval uses deterministic lexical matching for explicit terms; it is not
          labeled as BM25 here.
        </div>
        <div className="rounded-token border border-border bg-surface p-4 text-sm text-muted">
          Hybrid search combines dense and sparse ranks with weighted reciprocal rank
          fusion. Fused score is ranking evidence, not a probability or confidence.
        </div>
      </div>
    </Section>
  );
}
