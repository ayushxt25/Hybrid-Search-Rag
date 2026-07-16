import { cn } from "../../lib/utils/cn";

export type TabItem = { id: string; label: string };

type TabsProps = {
  items: TabItem[];
  value: string;
  onChange: (value: string) => void;
};

export function Tabs({ items, value, onChange }: TabsProps) {
  return (
    <div
      role="tablist"
      className="inline-flex rounded-token border border-border bg-background p-1"
    >
      {items.map((item) => (
        <button
          key={item.id}
          role="tab"
          aria-selected={value === item.id}
          onClick={() => onChange(item.id)}
          className={cn(
            "rounded-md px-3 py-1.5 text-sm text-muted",
            value === item.id && "bg-elevated text-primary",
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
