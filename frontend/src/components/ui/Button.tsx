import { forwardRef } from "react";

import { cn } from "../../lib/utils/cn";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  isLoading?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant = "secondary", isLoading, children, disabled, ...props },
    ref,
  ) => (
    <button
      ref={ref}
      disabled={disabled || isLoading}
      className={cn(
        "inline-flex h-10 items-center justify-center gap-2 rounded-token px-4 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-accent text-slate-950 hover:bg-accent/90",
        variant === "secondary" &&
          "border border-border bg-elevated text-primary hover:bg-elevated/80",
        variant === "ghost" && "text-secondary hover:bg-elevated",
        className,
      )}
      {...props}
    >
      {isLoading ? "Loading..." : children}
    </button>
  ),
);
Button.displayName = "Button";
