import { ChevronLeft, ChevronRight } from "lucide-react";
import { NavLink } from "react-router-dom";

import { navigation } from "../../routes/navigation";
import { IconButton } from "../ui/IconButton";
import { cn } from "../../lib/utils/cn";

type SidebarProps = {
  collapsed: boolean;
  onToggle: () => void;
};

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={cn(
        "hidden border-r border-border bg-surface transition-all lg:flex lg:flex-col",
        collapsed ? "lg:w-20" : "lg:w-72",
      )}
    >
      <div className="flex h-16 items-center justify-between border-b border-border px-4">
        <div className={cn("min-w-0", collapsed && "sr-only")}>
          <div className="text-sm font-semibold">Hybrid Search Studio</div>
          <div className="text-xs text-muted">RAG developer console</div>
        </div>
        <IconButton
          label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          onClick={onToggle}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </IconButton>
      </div>
      <nav aria-label="Primary navigation" className="flex-1 space-y-1 p-3">
        {navigation.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-token px-3 py-2 text-sm text-secondary hover:bg-elevated",
                isActive && "bg-elevated text-primary ring-1 ring-border",
                collapsed && "justify-center px-2",
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon aria-hidden className="h-4 w-4 shrink-0" />
                <span className={cn(collapsed && "sr-only")}>{item.label}</span>
                {isActive ? <span className="sr-only">Current page</span> : null}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-border p-4 text-xs text-muted">
        <span className={cn(collapsed && "sr-only")}>Foundation v0.1</span>
      </div>
    </aside>
  );
}
