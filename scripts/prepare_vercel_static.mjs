import { cp, rm } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const source = resolve(root, "frontend", "dist");
const destination = resolve(root, "public");

await rm(destination, { force: true, recursive: true });
await cp(source, destination, { recursive: true });
