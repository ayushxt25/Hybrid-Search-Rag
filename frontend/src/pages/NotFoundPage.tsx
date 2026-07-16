import { Link } from "react-router-dom";

import { EmptyState } from "../components/feedback/States";

export function NotFoundPage() {
  return (
    <div className="mx-auto max-w-2xl">
      <EmptyState
        title="Page not found"
        description="The requested Studio route does not exist."
      />
      <div className="mt-4 flex justify-center">
        <Link
          className="rounded-token bg-accent px-4 py-2 text-sm font-medium text-slate-950 hover:bg-accent/90"
          to="/overview"
        >
          Return to overview
        </Link>
      </div>
    </div>
  );
}
