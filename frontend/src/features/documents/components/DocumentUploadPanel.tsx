import { FileUp, X } from "lucide-react";
import { useRef, useState } from "react";

import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { IconButton } from "../../../components/ui/IconButton";
import { acceptAttribute, maxUploadLabel, validateUploadFile } from "../utils";

type DocumentUploadPanelProps = {
  disabled: boolean;
  isUploading: boolean;
  error: string | null;
  onUpload: (file: File) => void;
};

export function DocumentUploadPanel({
  disabled,
  isUploading,
  error,
  onUpload,
}: DocumentUploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<string | null>(null);
  const selectedError = validation ?? error;

  function choose(nextFile: File | null) {
    setFile(nextFile);
    setValidation(validateUploadFile(nextFile));
  }

  function submit() {
    const nextError = validateUploadFile(file);
    setValidation(nextError);
    if (!file || nextError) return;
    onUpload(file);
  }

  return (
    <Card>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="font-semibold">Upload document</h2>
          <p className="mt-2 text-sm text-muted">
            TXT, Markdown, PDF, or DOCX. One file at a time, up to {maxUploadLabel}.
          </p>
        </div>
        <FileUp className="h-5 w-5 text-accent" />
      </div>
      <button
        type="button"
        className="mt-4 flex min-h-36 w-full flex-col items-center justify-center rounded-token border border-dashed border-border bg-background p-5 text-center text-sm text-secondary transition hover:bg-elevated focus:outline-none focus:ring-2 focus:ring-focus"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        onDrop={(event) => {
          event.preventDefault();
          choose(event.dataTransfer.files.item(0));
        }}
        onDragOver={(event) => event.preventDefault()}
        disabled={disabled || isUploading}
        aria-describedby="document-upload-help document-upload-error"
      >
        <FileUp className="mb-3 h-6 w-6 text-muted" />
        Drop a document here or open the file picker
      </button>
      <input
        ref={inputRef}
        aria-label="Choose document file"
        className="sr-only"
        type="file"
        accept={acceptAttribute}
        onChange={(event) => choose(event.target.files?.item(0) ?? null)}
      />
      <p id="document-upload-help" className="mt-3 text-xs text-muted">
        Upload starts only after you choose Upload. Progress is shown as a working
        state, not a percentage.
      </p>
      {file ? (
        <div className="mt-4 flex items-center justify-between gap-3 rounded-token border border-border bg-background p-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted">{Math.ceil(file.size / 1024)} KB</p>
          </div>
          <IconButton label="Clear selected file" onClick={() => choose(null)}>
            <X className="h-4 w-4" />
          </IconButton>
        </div>
      ) : null}
      {selectedError ? (
        <p id="document-upload-error" role="alert" className="mt-3 text-sm text-danger">
          {selectedError}
        </p>
      ) : null}
      <div className="mt-5 flex gap-2">
        <Button
          variant="primary"
          onClick={submit}
          disabled={disabled || Boolean(validateUploadFile(file))}
          isLoading={isUploading}
        >
          Upload
        </Button>
        <Button variant="ghost" onClick={() => choose(null)} disabled={isUploading}>
          Reset
        </Button>
      </div>
    </Card>
  );
}
