import { useSyncExternalStore } from "react";

import {
  getSessionApiKeySnapshot,
  subscribeSessionApiKey,
} from "../../lib/api/client";

export function useSessionApiKeyStatus() {
  return useSyncExternalStore(
    subscribeSessionApiKey,
    getSessionApiKeySnapshot,
    getSessionApiKeySnapshot,
  );
}
