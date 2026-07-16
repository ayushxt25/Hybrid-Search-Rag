export function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="overflow-auto rounded-token border border-border bg-background p-4 text-sm text-secondary">
      <code>{children}</code>
    </pre>
  );
}
