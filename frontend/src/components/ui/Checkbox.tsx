import { forwardRef } from "react";

export const Checkbox = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    type="checkbox"
    className={
      className ?? "h-4 w-4 rounded border-border bg-background accent-sky-400"
    }
    {...props}
  />
));
Checkbox.displayName = "Checkbox";
