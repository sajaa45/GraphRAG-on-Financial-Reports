const KEY = "equitygraph.backend_url";

export const DEFAULT_BACKEND_URL = "http://localhost:8000";

export function getBackendUrl(): string {
  if (typeof window === "undefined") return DEFAULT_BACKEND_URL;
  return localStorage.getItem(KEY) || DEFAULT_BACKEND_URL;
}

export function setBackendUrl(url: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, url.replace(/\/$/, ""));
}

export const PIPELINE_STEPS = [
  { id: 1, name: "parse_document", label: "Parse & convert document", description: "Convert the uploaded HTML/PDF into structured markdown sections." },
  { id: 2, name: "extract_target_entities", label: "Extract target company entities & build KG", description: "LLM extracts metrics, risks, and operating segments into Neo4j." },
  { id: 3, name: "find_peers", label: "Identify SIC code & find peers via EDGAR", description: "Look up the SIC code and pull comparable filers from EDGAR." },
  { id: 4, name: "retrieve_peer_metrics", label: "Retrieve peer metrics via XBRL", description: "Fetch financial metrics for each peer from XBRL filings." },
  { id: 5, name: "retrieve_peer_risks", label: "Retrieve peer risk factors from HTM filings", description: "Pull risk factor sections from peer HTM filings." },
  { id: 6, name: "build_peer_kg", label: "Build & populate peer knowledge graph", description: "Materialize peer entities and relationships in the graph." },
] as const;
