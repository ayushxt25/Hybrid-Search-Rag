import { Check, Copy } from "lucide-react";
import { useState } from "react";

import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { contentTypeLabel, truncateId } from "../../documents/utils";
import type { IndexedDocumentSummary } from "../../documents/types";
import type { AnswerCitation } from "../types";

type SourceCardProps = {
  citation: AnswerCitation;
  documents: IndexedDocumentSummary[];
  selected: boolean;
};

export function SourceCard({ citation, documents, selected }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const document = documents.find((item) => item.document_id === citation.document_id);
  const filename = citation.file_name || document?.filename || "Unknown document";
  const sourceText = citation.heading ?? "Source excerpt was not returned by the API.";
  const isLong = sourceText.length > 220;
  const visibleText =
    expanded || !isLong ? sourceText : `${sourceText.slice(0, 220)}...`;

  async function copy(value: string, label: string) {
    await navigator.clipboard?.writeText(value);
    setCopied(label);
    window.setTimeout(() => setCopied(null), 1600);
  }

  return (
    <article
      id={`source-${citation.source_number}`}
      className={[
        "rounded-token border bg-surface p-4 shadow-token",
        selected ? "border-accent ring-2 ring-focus" : "border-border",
      ].join(" ")}
      aria-label={`Source ${citation.source_number}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="accent">Source {citation.source_number}</Badge>
            <h3 className="font-semibold">{filename}</h3>
            <Badge tone="neutral">
              {contentTypeLabel(document?.content_type ?? null)}
            </Badge>
          </div>
          <p className="mt-2 text-sm text-muted">
            Document {truncateId(citation.document_id)}
            {citation.page_number ? ` · page ${citation.page_number}` : ""}
          </p>
        </div>
        <div className="flex gap-2">
          <IconButton
            label="Copy source document ID"
            onClick={() => copy(citation.document_id, "id")}
          >
            {copied === "id" ? <Check aria-hidden /> : <Copy aria-hidden />}
          </IconButton>
          <IconButton label="Copy source text" onClick={() => copy(sourceText, "text")}>
            {copied === "text" ? <Check aria-hidden /> : <Copy aria-hidden />}
          </IconButton>
        </div>
      </div>
      <p className="mt-3 whitespace-pre-wrap break-words rounded-token border border-border bg-background p-3 text-sm text-secondary">
        {visibleText}
      </p>
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
      <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted">
        {citation.heading && <span>Heading returned</span>}
        {selected && <span>Selected source</span>}
        {copied && <span role="status">Copied.</span>}
      </div>
    </article>
  );
}
