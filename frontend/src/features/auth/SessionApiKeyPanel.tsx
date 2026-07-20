import { useState } from "react";

import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { Input } from "../../components/ui/Input";
import {
  clearSessionApiKey,
  setSessionApiKey,
} from "../../lib/api/client";
import { useSessionApiKeyStatus } from "./sessionApiKeyHooks";

export function SessionApiKeyPanel({ onChange }: { onChange?: () => void }) {
  const [value, setValue] = useState("");
  const hasKey = useSessionApiKeyStatus();

  function save(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSessionApiKey(value);
    setValue("");
    onChange?.();
  }

  function clear() {
    clearSessionApiKey();
    setValue("");
    onChange?.();
  }

  return (
    <Card>
      <h2 className="font-semibold">Session API key</h2>
      <p className="mt-2 text-sm text-muted">
        Kept only for the current browser tab/session and sent as the X-API-Key
        header. The key is never displayed after saving.
      </p>
      <form className="mt-4 space-y-3" onSubmit={save}>
        <label htmlFor="session-api-key" className="block text-sm text-secondary">
          Session API key
        </label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            id="session-api-key"
            type="password"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder={hasKey ? "Key set for this session" : "Paste key"}
            autoComplete="off"
          />
          <Button type="submit" disabled={!value.trim()}>
            {hasKey ? "Update" : "Save"}
          </Button>
          <Button type="button" variant="ghost" onClick={clear} disabled={!hasKey}>
            Clear
          </Button>
        </div>
      </form>
    </Card>
  );
}
