import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

import { useHealth } from "./useHealth";
import { StatusBadge } from "../../components/ui/StatusBadge";

export type HealthUiState = "checking" | "connected" | "unavailable";

function healthUiState(health: {
  isLoading: boolean;
  isError: boolean;
}): HealthUiState {
  if (health.isLoading) return "checking";
  if (health.isError) return "unavailable";
  return "connected";
}

export function HealthStatusBadge({ state }: { state: HealthUiState }) {
  if (state === "checking") {
    return (
      <StatusBadge
        tone="muted"
        icon={Loader2}
        label="Checking"
        className="animate-pulse"
      />
    );
  }

  if (state === "unavailable") {
    return <StatusBadge tone="danger" icon={AlertCircle} label="Unavailable" />;
  }

  return <StatusBadge tone="success" icon={CheckCircle2} label="Connected" />;
}

export function HealthStatus() {
  const health = useHealth();
  return <HealthStatusBadge state={healthUiState(health)} />;
}
