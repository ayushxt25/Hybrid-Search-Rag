type ToastProps = {
  message: string;
};

export function Toast({ message }: ToastProps) {
  return (
    <div
      role="status"
      className="rounded-token border border-border bg-elevated px-4 py-3 text-sm text-secondary shadow-token"
    >
      {message}
    </div>
  );
}
