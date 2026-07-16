import { forwardRef } from "react";

import { cn } from "../../lib/utils/cn";

export const Select = forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "h-10 w-full rounded-token border border-border bg-background px-3 text-sm text-primary disabled:opacity-50",
      className,
    )}
    {...props}
  />
));
Select.displayName = "Select";
