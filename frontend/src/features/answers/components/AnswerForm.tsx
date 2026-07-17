import { RotateCcw, Send } from "lucide-react";

import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Input } from "../../../components/ui/Input";
import { Textarea } from "../../../components/ui/Textarea";
import type { GroundedAnswerPayload } from "../types";
import { validateAnswerRequest } from "../validation";

type AnswerFormProps = {
  question: string;
  limit: number;
  candidateLimit: number;
  documentIds: string[];
  contentTypes: string[];
  backendAvailable: boolean;
  isAsking: boolean;
  onQuestionChange: (value: string) => void;
  onLimitChange: (value: number) => void;
  onCandidateLimitChange: (value: number) => void;
  onSubmit: (payload: GroundedAnswerPayload) => void;
  onReset: () => void;
};

export function AnswerForm({
  question,
  limit,
  candidateLimit,
  documentIds,
  contentTypes,
  backendAvailable,
  isAsking,
  onQuestionChange,
  onLimitChange,
  onCandidateLimitChange,
  onSubmit,
  onReset,
}: AnswerFormProps) {
  const validationError = validateAnswerRequest({
    question,
    limit,
    candidateLimit,
    documentIds,
    contentTypes,
  });
  const disabled = isAsking || !backendAvailable || Boolean(validationError);

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled) return;
    onSubmit({
      question: question.trim(),
      limit,
      candidate_limit: candidateLimit,
      document_ids: documentIds,
      content_types: contentTypes,
    });
  }

  return (
    <Card>
      <form onSubmit={submit} className="space-y-5">
        <div>
          <label htmlFor="grounded-question" className="text-sm font-medium">
            Question
          </label>
          <Textarea
            id="grounded-question"
            value={question}
            onChange={(event) => onQuestionChange(event.target.value)}
            placeholder="Ask a question that should be answered from indexed evidence."
            aria-describedby="answer-question-help answer-form-error"
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <p id="answer-question-help" className="mt-2 text-xs text-muted">
            Press Enter to ask. Use Shift+Enter for a new line.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="text-sm font-medium">
            Result limit
            <Input
              className="mt-2"
              type="number"
              min={1}
              max={50}
              value={limit}
              onChange={(event) => onLimitChange(Number(event.target.value))}
            />
          </label>
          <label className="text-sm font-medium">
            Candidate limit
            <Input
              className="mt-2"
              type="number"
              min={1}
              max={100}
              value={candidateLimit}
              onChange={(event) => onCandidateLimitChange(Number(event.target.value))}
            />
          </label>
        </div>

        <p id="answer-form-error" role="alert" className="min-h-5 text-sm text-danger">
          {!backendAvailable
            ? "Backend is unavailable."
            : validationError
              ? validationError
              : ""}
        </p>

        <div className="flex flex-wrap gap-3">
          <Button
            type="submit"
            variant="primary"
            isLoading={isAsking}
            disabled={disabled}
          >
            <Send aria-hidden className="h-4 w-4" />
            Ask
          </Button>
          <Button type="button" variant="secondary" onClick={onReset}>
            <RotateCcw aria-hidden className="h-4 w-4" />
            Reset
          </Button>
        </div>
      </form>
    </Card>
  );
}
