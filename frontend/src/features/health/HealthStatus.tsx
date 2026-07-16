import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

import { useHealth } from "./useHealth";
import { StatusBadge } from "../../components/ui/StatusBadge";

export function HealthStatus() {
  const health = useHealth();

  if (health.isLoading) {
    return (
      <StatusBadge
        tone="muted"
        icon={Loader2}
        label="Checking"
        className="animate-pulse"
      />
    );
  }

  if (health.isError) {
    return <StatusBadge tone="danger" icon={AlertCircle} label="Unavailable" />;
  }

  return <StatusBadge tone="success" icon={CheckCircle2} label="Connected" />;
}
