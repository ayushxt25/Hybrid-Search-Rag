import { X } from "lucide-react";
import { useEffect } from "react";
import { NavLink } from "react-router-dom";

import { navigation } from "../../routes/navigation";
import { IconButton } from "../ui/IconButton";
import { cn } from "../../lib/utils/cn";

type MobileNavigationProps = {
  open: boolean;
  onClose: () => void;
};

export function MobileNavigation({ open, onClose }: MobileNavigationProps) {
  useEffect(() => {
    if (!open) return;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 lg:hidden"
      role="dialog"
      aria-modal="true"
      aria-label="Mobile navigation"
    >
      <button
        className="absolute inset-0 bg-black/60"
        aria-label="Close navigation"
        onClick={onClose}
      />
      <div className="relative flex h-full w-80 max-w-[85vw] flex-col border-r border-border bg-surface shadow-token">
        <div className="flex h-16 items-center justify-between border-b border-border px-4">
          <div>
            <div className="text-sm font-semibold">Hybrid Search Studio</div>
            <div className="text-xs text-muted">RAG developer console</div>
          </div>
          <IconButton label="Close navigation" onClick={onClose}>
            <X className="h-4 w-4" />
          </IconButton>
        </div>
        <nav aria-label="Mobile primary navigation" className="space-y-1 p-3">
          {navigation.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onClose}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-token px-3 py-2 text-sm text-secondary",
                  isActive && "bg-elevated text-primary",
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon aria-hidden className="h-4 w-4" />
                  {item.label}
                  {isActive ? <span className="sr-only">Current page</span> : null}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  );
}
