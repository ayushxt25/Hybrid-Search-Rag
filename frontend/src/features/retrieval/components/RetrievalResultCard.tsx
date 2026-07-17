import { Check, Copy } from "lucide-react";
import { useMemo, useState } from "react";

import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { contentTypeLabel, truncateId } from "../../documents/utils";
import type { IndexedDocumentSummary } from "../../documents/types";
import type { SearchResult } from "../types";
import { ScoreDiagnostics } from "./ScoreDiagnostics";

type RetrievalResultCardProps = {
  result: SearchResult;
  rank: number;
  documents: IndexedDocumentSummary[];
};

export function RetrievalResultCard({
  result,
  rank,
  documents,
}: RetrievalResultCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const document = documents.find((item) => item.document_id === result.document_id);
  const filename = result.file_name || document?.filename || "Unknown document";
  const contentLabel = useMemo(() => {
    if (document?.content_type) return contentTypeLabel(document.content_type);
    return result.file_extension?.replace(/^\./, "").toUpperCase() || "Unknown";
  }, [document?.content_type, result.file_extension]);
  const isLong = result.text.length > 520;
  const visibleText =
    expanded || !isLong ? result.text : `${result.text.slice(0, 520)}...`;

  async function copy(value: string, label: string) {
    await navigator.clipboard?.writeText(value);
    setCopied(label);
    window.setTimeout(() => setCopied(null), 1600);
  }

  return (
    <article className="rounded-token border border-border bg-surface p-4 shadow-token">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="accent">Rank {rank}</Badge>
            <h3 className="font-semibold">{filename}</h3>
            <Badge tone="neutral">{contentLabel}</Badge>
          </div>
          <p className="mt-2 text-sm text-muted">
            Document {truncateId(result.document_id)} · chunk {result.chunk_index}
            {result.section_index !== undefined && ` · section ${result.section_index}`}
            {result.page_number && ` · page ${result.page_number}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <IconButton
            label="Copy document ID"
            onClick={() => copy(result.document_id, "id")}
          >
            {copied === "id" ? <Check aria-hidden /> : <Copy aria-hidden />}
          </IconButton>
          <IconButton label="Copy chunk text" onClick={() => copy(result.text, "text")}>
            {copied === "text" ? <Check aria-hidden /> : <Copy aria-hidden />}
          </IconButton>
        </div>
      </div>

      {result.heading && <p className="mt-3 text-sm font-medium">{result.heading}</p>}
      <pre className="mt-3 whitespace-pre-wrap break-words rounded-token border border-border bg-background p-4 text-sm leading-6 text-secondary">
        {visibleText}
      </pre>
      {isLong && (
        <Button
          type="button"
          variant="ghost"
          className="mt-2 h-8 px-2"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Show less" : "Show more"}
        </Button>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>Score {result.score.toFixed(6)}</span>
        <span>
          Words {result.start_word}-{result.end_word} ({result.word_count})
        </span>
        {copied && <span role="status">Copied.</span>}
      </div>
      {result.score_diagnostics && (
        <div className="mt-4">
          <ScoreDiagnostics diagnostics={result.score_diagnostics} />
        </div>
      )}
    </article>
  );
}
