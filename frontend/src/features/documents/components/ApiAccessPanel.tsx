import { useState } from "react";

import {
  clearSessionApiKey,
  hasSessionApiKey,
  setSessionApiKey,
} from "../../../lib/api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Input } from "../../../components/ui/Input";

export function ApiAccessPanel({ onChange }: { onChange: () => void }) {
  const [value, setValue] = useState("");
  const [hasKey, setHasKey] = useState(hasSessionApiKey());

  function save() {
    setSessionApiKey(value);
    setHasKey(hasSessionApiKey());
    setValue("");
    onChange();
  }

  function clear() {
    clearSessionApiKey();
    setHasKey(false);
    setValue("");
    onChange();
  }

  return (
    <Card>
      <h2 className="font-semibold">API access</h2>
      <p className="mt-2 text-sm text-muted">
        Optional API key for this page session only. It is kept in memory and is never
        stored.
      </p>
      <label htmlFor="documents-api-key" className="mt-4 block text-sm text-secondary">
        API key
      </label>
      <div className="mt-2 flex gap-2">
        <Input
          id="documents-api-key"
          type="password"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={hasKey ? "Key set for this session" : "Optional"}
          autoComplete="off"
        />
        <Button onClick={save} disabled={!value.trim()}>
          Set
        </Button>
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 text-sm text-muted">
        <span>{hasKey ? "A session key is active." : "No session key set."}</span>
        <Button variant="ghost" onClick={clear} disabled={!hasKey}>
          Clear
        </Button>
      </div>
    </Card>
  );
}
