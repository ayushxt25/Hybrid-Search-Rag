export type MetadataItem = { label: string; value: React.ReactNode };

export function MetadataList({ items }: { items: MetadataItem[] }) {
  return (
    <dl className="grid gap-3 sm:grid-cols-2">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded-token border border-border bg-background p-3"
        >
          <dt className="text-xs uppercase tracking-wide text-muted">{item.label}</dt>
          <dd className="mt-1 text-sm text-secondary">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
