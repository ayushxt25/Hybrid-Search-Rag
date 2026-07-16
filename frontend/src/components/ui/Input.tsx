import { forwardRef } from "react";

import { cn } from "../../lib/utils/cn";

export const Input = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-10 w-full rounded-token border border-border bg-background px-3 text-sm text-primary placeholder:text-muted disabled:opacity-50",
      className,
    )}
    {...props}
  />
));
Input.displayName = "Input";
