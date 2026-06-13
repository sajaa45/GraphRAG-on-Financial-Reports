const KEY = "equitygraph.backend_url";

export const DEFAULT_BACKEND_URL = "http://192.168.100.9:8080";

export function getBackendUrl(): string {
  if (typeof window === "undefined") return DEFAULT_BACKEND_URL;
  const stored = localStorage.getItem(KEY);
  // If the stored value is the old default (localhost:8000), replace it.
  if (!stored || stored === "http://localhost:8000") {
    localStorage.setItem(KEY, DEFAULT_BACKEND_URL);
    return DEFAULT_BACKEND_URL;
  }
  return stored;
}

export function setBackendUrl(url: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, url.replace(/\/$/, ""));
}

export const PIPELINE_STEPS = [
  { id: 1, name: "parse_document",         label: "Parse & convert document",          description: "Convert the uploaded filing into structured sections." },
  { id: 2, name: "extract_target_entities", label: "Extract target entities",           description: "Extract financial metrics, risks, and operating entities from the filing." },
  { id: 3, name: "build_target_kg",         label: "Build target knowledge graph",      description: "Write the extracted target company nodes and relationships into Neo4j." },
  { id: 4, name: "retrieve_peer_data",      label: "Find peers & retrieve data",        description: "Identify peer companies via SIC code and fetch their metrics and risk filings." },
  { id: 5, name: "build_peer_kg",           label: "Build peer knowledge graph",        description: "Write peer company metrics and risks into the knowledge graph." },
] as const;
