import { Badge } from "../../../components/ui/Badge";
import type { AnswerCitation } from "../types";

type CitationListProps = {
  citations: AnswerCitation[];
  markers: number[];
  selectedSource: number | null;
  onSelect: (sourceNumber: number) => void;
};

export function CitationList({
  citations,
  markers,
  selectedSource,
  onSelect,
}: CitationListProps) {
  if (!citations.length) return null;
  return (
    <section aria-labelledby="citation-list-heading" className="space-y-3">
      <h3 id="citation-list-heading" className="font-semibold">
        Citations
      </h3>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation) => (
          <button
            key={citation.source_number}
            type="button"
            onClick={() => onSelect(citation.source_number)}
            className="rounded-full focus:outline-none focus:ring-2 focus:ring-focus"
            aria-pressed={selectedSource === citation.source_number}
          >
            <Badge
              tone={selectedSource === citation.source_number ? "accent" : "neutral"}
            >
              [Source {citation.source_number}]
            </Badge>
          </button>
        ))}
      </div>
      <p className="text-xs text-muted">
        Markers returned in answer:{" "}
        {markers.length
          ? markers.map((marker) => `[Source ${marker}]`).join(", ")
          : "none"}
      </p>
    </section>
  );
}
