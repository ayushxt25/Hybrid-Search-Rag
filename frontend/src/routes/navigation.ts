import {
  Activity,
  Database,
  FileQuestion,
  Gauge,
  MessageSquareQuote,
} from "lucide-react";

export const navigation = [
  { label: "Overview", path: "/overview", icon: Gauge },
  { label: "Documents", path: "/documents", icon: Database },
  { label: "Retrieval Playground", path: "/retrieval", icon: Activity },
  { label: "Grounded Answers", path: "/answers", icon: MessageSquareQuote },
  { label: "System Health", path: "/system", icon: FileQuestion },
];
