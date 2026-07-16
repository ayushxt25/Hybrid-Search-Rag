import { cn } from "../../lib/utils/cn";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-token border border-border bg-surface p-5 shadow-token",
        className,
      )}
      {...props}
    />
  );
}
