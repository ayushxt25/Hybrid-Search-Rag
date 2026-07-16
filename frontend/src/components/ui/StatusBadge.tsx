import type { LucideIcon } from "lucide-react";

import { Badge } from "./Badge";
import { cn } from "../../lib/utils/cn";

type StatusBadgeProps = {
  label: string;
  icon?: LucideIcon;
  tone?: "muted" | "success" | "warning" | "danger";
  className?: string;
};

export function StatusBadge({
  label,
  icon: Icon,
  tone = "muted",
  className,
}: StatusBadgeProps) {
  const badgeTone = tone === "muted" ? "neutral" : tone;
  return (
    <Badge tone={badgeTone} className={cn("gap-1.5", className)}>
      {Icon ? <Icon aria-hidden className="h-3.5 w-3.5" /> : null}
      {label}
    </Badge>
  );
}
