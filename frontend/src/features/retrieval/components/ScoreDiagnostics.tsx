import type { BranchScoreDiagnostic, RetrievalScoreDiagnostic } from "../types";

function formatNumber(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(6);
}

function BranchDiagnostic({
  label,
  branch,
}: {
  label: string;
  branch: BranchScoreDiagnostic;
}) {
  return (
    <div className="rounded-token border border-border bg-background p-3">
      <h4 className="text-sm font-medium">{label}</h4>
      <dl className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <dt className="text-muted">Raw score</dt>
        <dd className="text-right text-secondary">
          {branch.raw_score === null
            ? "Not present in branch"
            : formatNumber(branch.raw_score)}
        </dd>
        <dt className="text-muted">Rank</dt>
        <dd className="text-right text-secondary">
          {branch.rank === null ? "Not present in branch" : branch.rank}
        </dd>
        <dt className="text-muted">Weight</dt>
        <dd className="text-right text-secondary">{formatNumber(branch.weight)}</dd>
        <dt className="text-muted">RRF contribution</dt>
        <dd className="text-right text-secondary">
          {formatNumber(branch.rrf_contribution)}
        </dd>
      </dl>
    </div>
  );
}

export function ScoreDiagnostics({
  diagnostics,
}: {
  diagnostics: RetrievalScoreDiagnostic;
}) {
  return (
    <details className="rounded-token border border-border bg-elevated p-4">
      <summary className="cursor-pointer text-sm font-medium">
        Score diagnostics
      </summary>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <BranchDiagnostic label="Dense branch" branch={diagnostics.dense} />
        <BranchDiagnostic label="Sparse branch" branch={diagnostics.sparse} />
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-2 rounded-token border border-border bg-background p-3 text-xs">
        <dt className="text-muted">Fused score</dt>
        <dd className="text-right text-secondary">
          {formatNumber(diagnostics.fused_score)}
        </dd>
        <dt className="text-muted">Fused rank</dt>
        <dd className="text-right text-secondary">{diagnostics.fused_rank}</dd>
      </dl>
      <p className="mt-3 text-xs leading-5 text-muted">
        Dense and sparse raw scores are not directly comparable. Weighted RRF combines
        ranks, not raw score magnitudes. Fused score is not a probability or confidence
        percentage. A result can appear in only one branch and still be included.
      </p>
    </details>
  );
}
