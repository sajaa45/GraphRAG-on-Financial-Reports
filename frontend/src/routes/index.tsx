import React, { useCallback, useEffect, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  Upload,
  Play,
  FileText,
  Settings,
  RotateCcw,
  AlertCircle,
  Send,
  ShieldCheck,
  Sparkles,
  Building2,
  ChevronDown,
  Moon,
  Palette,
  Sun,
  Database,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { StepItem, type StepData, type StepStatus } from "@/components/StepItem";
import {
  PIPELINE_STEPS,
  getBackendUrl,
  setBackendUrl,
  DEFAULT_BACKEND_URL,
} from "@/lib/pipeline-config";

export const Route = createFileRoute("/")({
  component: Index,
});

type PhaseStatus = "idle" | "running" | "complete" | "failed";

interface CitationInfo {
  type: "risk" | "metric";
  company: string;
  role: "target" | "peer";
  document_url: string | null;
  summary: string;
  source_page?: number | string | null;
  section_title?: string | null;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning_trace?: string | null;
  citations?: Record<string, CitationInfo>;
  trace?: { step: string; status: string }[];
  evaluation?: EvaluationResult | null;
  evaluating?: boolean;
  evalType?: string;
}

interface EvaluationResult {
  test_type: string;
  rows: { dimension: string; score: number; note: string }[];
  weighted: number;
}

interface CompanyOption {
  name: string;
  cik?: string | null;
}

type ColorMode = "dark" | "light";
type PaletteName = "verdant" | "sage" | "sky" | "rose" | "lavender";

const PALETTES: { value: PaletteName; label: string; previewClass: string }[] = [
  { value: "verdant", label: "Verdant", previewClass: "bg-palette-verdant" },
  { value: "sage", label: "Sage", previewClass: "bg-palette-sage" },
  { value: "sky", label: "Sky", previewClass: "bg-palette-sky" },
  { value: "rose", label: "Rose", previewClass: "bg-palette-rose" },
  { value: "lavender", label: "Lavender", previewClass: "bg-palette-lavender" },
];

const EVAL_TESTS: { value: string; label: string }[] = [
  { value: "answer_relevancy",           label: "Answer Relevancy" },
  { value: "context_precision",          label: "Context Precision" },
  { value: "answer_source_traceability", label: "Source Traceability" },
  { value: "target_validation",          label: "Target Extraction" },
  { value: "risk_peers_validation",      label: "Peer Risk Validation" },
  { value: "overall_score",              label: "Overall Score" },
];

function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

const QA_TRACE_STEPS = [
  "cypher_translation",
  "graph_traversal",
  "context_retrieval",
  "answer_generation",
  "citation_attachment",
];

function BrandLogo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 1024 1024" className={className} xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      {/* monitor outer frame */}
      <path fillOpacity=".18" fill="currentColor" d="M912.9 732.5c-47 0-86.3 33.5-95.3 77.8H701.9c-11.7 0-20.2-2.3-23.2-6.3-2-2.7-2-7-2-9v-51.1h109.9c37.5 0 67.9-30.5 67.9-67.9V235.2c0-37.5-30.5-67.9-67.9-67.9h-706c-37.5 0-67.9 30.5-67.9 67.9v440.7c0 37.5 30.5 67.9 67.9 67.9h557.2V795c0 4.8 0 19.3 9.9 32.4 10.9 14.4 29.1 21.7 54.2 21.7h308.2v-19.4c0-53.6-43.6-97.2-97.2-97.2z"/>
      {/* screen bezel */}
      <path fillOpacity=".55" fill="currentColor" d="M815.6 675.9V235.2c0-16.1-13.1-29.1-29.1-29.1H80.6c-16.1 0-29.1 13.1-29.1 29.1v440.7c0 16.1 13.1 29.1 29.1 29.1h705.9c16.1.1 29.1-13 29.1-29.1zm-43.9-42c0 16-13.1 29.1-29.1 29.1h-618c-16 0-29.1-13.1-29.1-29.1V277.3c0-16 13.1-29.1 29.1-29.1h618c16 0 29.1 13.1 29.1 29.1v356.6z"/>
      {/* screen glass */}
      <path fillOpacity=".08" fill="currentColor" d="M742.6 248.2h-618c-16 0-29.1 13.1-29.1 29.1v356.6c0 16 13.1 29.1 29.1 29.1h618c16 0 29.1-13.1 29.1-29.1V277.3c0-16-13.1-29.1-29.1-29.1z"/>
      {/* pie chart outer ring */}
      <path fill="currentColor" d="M274 335.6h-9.7c-67.8 0-123 55.2-123 123s55.2 123 123 123 123-55.2 123-123v-9.7H274V335.6zm93.4 132.8c-4.9 52.6-49.3 93.9-103.1 93.9-57.1 0-103.6-46.5-103.6-103.6 0-53.8 41.3-98.2 93.9-103.1v112.8h112.8z"/>
      {/* pie slice */}
      <path fillOpacity=".5" fill="currentColor" d="M297.2 325.2v94.7h94.7c0-52.3-42.4-94.7-94.7-94.7z"/>
      {/* data rows — full */}
      <path fill="currentColor" d="M670.8 329.4H461.2c-5.4 0-9.7 4.3-9.7 9.7s4.3 9.7 9.7 9.7h209.7c5.4 0 9.7-4.3 9.7-9.7s-4.4-9.7-9.8-9.7z"/>
      <path fill="currentColor" d="M670.8 402.7H461.2c-5.4 0-9.7 4.3-9.7 9.7s4.3 9.7 9.7 9.7h209.7c5.4 0 9.7-4.3 9.7-9.7s-4.4-9.7-9.8-9.7z"/>
      {/* data rows — shorter */}
      <path fillOpacity=".55" fill="currentColor" d="M670.8 476.1H525.2c-5.4 0-9.7 4.3-9.7 9.7s4.3 9.7 9.7 9.7h145.6c5.4 0 9.7-4.3 9.7-9.7s-4.3-9.7-9.7-9.7z"/>
      <path fillOpacity=".55" fill="currentColor" d="M670.8 549.4H530.1c-5.4 0-9.7 4.3-9.7 9.7 0 5.4 4.3 9.7 9.7 9.7h140.7c5.4 0 9.7-4.3 9.7-9.7 0-5.4-4.3-9.7-9.7-9.7z"/>
      {/* stand base */}
      <path fillOpacity=".35" fill="currentColor" d="M594.7 800.4H272.4c-10.7 0-19.4 8.7-19.4 19.4s8.7 19.4 19.4 19.4h322.3c10.7 0 19.4-8.7 19.4-19.4 0-10.8-8.7-19.4-19.4-19.4z"/>
      {/* title bar dots */}
      <circle fillOpacity=".7" fill="currentColor" cx="150" cy="288.8" r="11.9"/>
      <circle fillOpacity=".7" fill="currentColor" cx="189" cy="288.8" r="11.9"/>
    </svg>
  );
}

function uid(): string {
  // crypto.randomUUID() requires a secure context (HTTPS/localhost) — use a
  // Math.random fallback so HTTP network access (e.g. 192.168.x.x) works too.
  try { return crypto.randomUUID(); } catch {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }
}

function Index() {
  // -------- Backend config --------
  const [backendUrl, setBackendUrlState] = useState(DEFAULT_BACKEND_URL);
  const [showSettings, setShowSettings] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("dark");
  const [palette, setPalette] = useState<PaletteName>("verdant");
  const [companies, setCompanies] = useState<CompanyOption[]>([]);
  const [selectedCompany, setSelectedCompany] = useState("");
  const [companiesLoading, setCompaniesLoading] = useState(false);
  const [ingestionMode, setIngestionMode] = useState<"upload" | "select">("upload");

  useEffect(() => {
    setBackendUrlState(getBackendUrl());
    const savedMode = localStorage.getItem("verdant_color_mode");
    const savedPalette = localStorage.getItem("verdant_palette");
    if (savedMode === "light" || savedMode === "dark") setColorMode(savedMode);
    if (PALETTES.some((item) => item.value === savedPalette)) {
      setPalette(savedPalette as PaletteName);
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.mode = colorMode;
    document.documentElement.dataset.palette = palette;
    document.documentElement.classList.toggle("dark", colorMode === "dark");
    localStorage.setItem("verdant_color_mode", colorMode);
    localStorage.setItem("verdant_palette", palette);
  }, [colorMode, palette]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${backendUrl}/`, { method: "GET" });
        if (!cancelled) setConnected(r.ok);
      } catch {
        if (!cancelled) setConnected(false);
      }
    })();
    return () => { cancelled = true; };
  }, [backendUrl]);

  const loadCompanies = useCallback(async () => {
    setCompaniesLoading(true);
    try {
      const response = await fetch(`${backendUrl}/companies`);
      if (!response.ok) throw new Error(`Companies request failed (${response.status})`);
      const data = (await response.json()) as { companies?: CompanyOption[] };
      const options = data.companies ?? [];
      setCompanies(options);
      setSelectedCompany((current) =>
        options.some((company) => company.name === current) ? current : options[0]?.name ?? "",
      );
    } catch {
      setCompanies([]);
      setSelectedCompany("");
    } finally {
      setCompaniesLoading(false);
    }
  }, [backendUrl]);

  useEffect(() => {
    void loadCompanies();
  }, [loadCompanies]);

  // On startup, ask the backend if Neo4j already has data.
  // This unlocks QA even when the pipeline was run outside the UI (e.g. via CLI).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${backendUrl}/qa/ready`);
        if (!cancelled && r.ok) {
          const data = await r.json();
          if (data.ready) {
            setPhase1Status((prev) => (prev === "idle" ? "complete" : prev));
          }
        }
      } catch { /* backend may not be up yet — ignore */ }
    })();
    return () => { cancelled = true; };
  }, [backendUrl]);

  // -------- Phase 1 state --------
  const [file, setFile] = useState<File | null>(null);
  const [fiscalYear, setFiscalYear] = useState("2024");
  const [previousFilename, setPreviousFilename] = useState<string | null>(null);
  const [savedJobId, setSavedJobId] = useState<string | null>(null);
  const [phase1Status, setPhase1Status] = useState<PhaseStatus>("idle");
  const [phase1Error, setPhase1Error] = useState<string | null>(null);
  const [steps, setSteps] = useState<StepData[]>(() =>
    PIPELINE_STEPS.map((s) => ({
      id: s.id,
      label: s.label,
      description: s.description,
      status: "pending" as StepStatus,
      logs: [],
    }))
  );

  // Restore persisted session state after hydration (client-only)
  useEffect(() => {
    try {
      setPreviousFilename(localStorage.getItem("verdant_filename"));
      setSavedJobId(localStorage.getItem("verdant_job_id"));
      if (localStorage.getItem("verdant_phase1_status") === "complete") {
        setPhase1Status("complete");
      }
      const saved = localStorage.getItem("verdant_steps");
      if (saved) {
        const parsed = JSON.parse(saved) as StepData[];
        if (parsed.length === PIPELINE_STEPS.length) setSteps(parsed);
      }
    } catch { /* ignore */ }
  }, []);
  const stepStartRef = useRef<Record<number, number>>({});
  const eventSourceRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  // -------- Phase 2: QA chat --------
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [withReasoning, setWithReasoning] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [showGraph, setShowGraph] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const phase2Ref = useRef<HTMLDivElement>(null);

  const resetSteps = useCallback(() => {
    setSteps(
      PIPELINE_STEPS.map((s) => ({
        id: s.id,
        label: s.label,
        description: s.description,
        status: "pending" as StepStatus,
        logs: [],
      })),
    );
    stepStartRef.current = {};
  }, []);

  // Attach SSE listener for a running job — shared by runPipeline and resumePipeline
  const startListening = useCallback((jobId: string, currentFile: File | null) => {
    const es = new EventSource(`${backendUrl}/pipeline/${jobId}/stream`);
    eventSourceRef.current = es;

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "pipeline_complete") {
          setPhase1Status("complete");
          setSteps((prev) => {
            try { localStorage.setItem("verdant_steps", JSON.stringify(prev)); } catch { /* ignore */ }
            return prev;
          });
          try {
            localStorage.setItem("verdant_phase1_status", "complete");
            if (currentFile) localStorage.setItem("verdant_filename", currentFile.name);
          } catch { /* ignore */ }
          es.close();
          return;
        }
        if (data.type === "pipeline_failed") {
          setPhase1Status("failed");
          setPhase1Error(data.error || "Pipeline failed");
          es.close();
          return;
        }
        const stepId = data.step as number;
        if (!stepId) return;
        setSteps((prev) =>
          prev.map((s) => {
            if (s.id !== stepId) return s;
            const next: StepData = { ...s };
            if (data.status === "running") {
              next.status = "running";
              if (!stepStartRef.current[stepId]) stepStartRef.current[stepId] = Date.now();
              if (data.message) next.logs = [...(s.logs || []), data.message];
            } else if (data.status === "done") {
              next.status = "done";
              next.summary = data.summary || data.message;
              const start = stepStartRef.current[stepId];
              if (start) next.durationMs = Date.now() - start;
            } else if (data.status === "failed") {
              next.status = "failed";
              next.summary = data.error || data.message || "Step failed";
            }
            return next;
          }),
        );
      } catch (e) {
        console.error("Failed to parse SSE event", e);
      }
    };

    es.onerror = () => {
      es.close();
      setPhase1Status((prev) => (prev === "running" ? "failed" : prev));
    };
  }, [backendUrl]);

  const runPipeline = useCallback(async () => {
    if (!file) return;
    try {
      localStorage.removeItem("verdant_phase1_status");
      localStorage.removeItem("verdant_steps");
      localStorage.removeItem("verdant_filename");
      localStorage.removeItem("verdant_job_id");
    } catch { /* ignore */ }
    setPreviousFilename(null);
    setSavedJobId(null);
    setPhase1Error(null);
    setPhase1Status("running");
    setMessages([]);
    resetSteps();

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("fiscal_year", fiscalYear);

      const r = await fetch(`${backendUrl}/pipeline/run`, { method: "POST", body: fd });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(`Upload failed (${r.status}): ${t}`);
      }
      const { job_id } = (await r.json()) as { job_id: string };
      try { localStorage.setItem("verdant_job_id", job_id); } catch { /* ignore */ }
      setSavedJobId(job_id);
      startListening(job_id, file);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setPhase1Error(msg);
      setPhase1Status("failed");
    }
  }, [file, fiscalYear, backendUrl, resetSteps, startListening]);

  const resumePipeline = useCallback(async () => {
    const firstIncompleteIdx = steps.findIndex((s) => s.status !== "done");
    const startFromStep = firstIncompleteIdx + 1;
    if (!savedJobId || firstIncompleteIdx < 0) {
      runPipeline();
      return;
    }

    // Keep done steps as-is; reset incomplete steps to pending.
    setSteps((prev) =>
      prev.map((s) => ({
        ...s,
        status:     Number(s.id) < startFromStep ? s.status     : ("pending" as StepStatus),
        logs:       Number(s.id) < startFromStep ? s.logs       : [],
        summary:    Number(s.id) < startFromStep ? s.summary    : undefined,
        durationMs: Number(s.id) < startFromStep ? s.durationMs : undefined,
      })),
    );
    stepStartRef.current = {};
    setPhase1Error(null);
    setPhase1Status("running");

    try {
      const r = await fetch(`${backendUrl}/pipeline/${savedJobId}/run`, {
        method: "POST",
      });
      if (!r.ok) throw new Error(`Resume failed (${r.status}): ${await r.text()}`);
      startListening(savedJobId, file);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setPhase1Error(msg);
      setPhase1Status("failed");
    }
  }, [steps, savedJobId, backendUrl, file, runPipeline, startListening]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  // Smooth scroll to QA when Phase 1 completes; clear stale graph cache so
  // the next "Explore graph" click fetches fresh data including new peers/metrics.
  useEffect(() => {
    if (phase1Status === "complete") {
      void loadCompanies();
      setGraphData(null);
      setShowGraph(false);
      setTimeout(() => phase2Ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 400);
    }
  }, [phase1Status, loadCompanies]);

  // Clear graph cache whenever the selected company changes.
  useEffect(() => {
    setGraphData(null);
    setShowGraph(false);
  }, [selectedCompany]);

  // -------- QA submit --------
  const pendingMsgId = useRef<string | null>(null);

  const askQuestion = () => {
    const q = question.trim();
    if (!q || asking) return;

    const userId = uid();
    const assistantId = uid();
    pendingMsgId.current = assistantId;

    setMessages((m) => [
      ...m,
      { id: userId, role: "user", content: q },
      { id: assistantId, role: "assistant", content: "", trace: QA_TRACE_STEPS.map((s) => ({ step: s, status: "pending" as const })) },
    ]);
    setQuestion("");
    setAsking(true);

    const url = backendUrl;
    const company = selectedCompany || null;
    const reasoning = withReasoning;

    fetch(`${url}/qa/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, reasoning, target_company: company }),
    })
      .then((r) => (r.ok ? r.json() : r.text().then((t) => Promise.reject(new Error(`${r.status}: ${t}`)))))
      .then((data) => {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: data.answer || "", citations: data.citations || {}, reasoning_trace: data.reasoning_trace || null, trace: msg.trace?.map((t) => ({ ...t, status: "done" as const })) }
              : msg,
          ),
        );
      })
      .catch((e: Error) => {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: e.message || "Request failed", trace: msg.trace?.map((t) => ({ ...t, status: "done" as const })) }
              : msg,
          ),
        );
      })
      .finally(() => {
        setAsking(false);
        pendingMsgId.current = null;
      });
  };

  // -------- Evaluation (per-message, user-chosen test type) --------
  const runEvaluation = useCallback(
    async (messageId: string, testType: string) => {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === messageId ? { ...msg, evaluating: true, evalType: testType } : msg,
        ),
      );

      let result: EvaluationResult | null = null;
      try {
        const r = await fetch(`${backendUrl}/eval/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ test_type: testType }),
        });
        if (r.ok) result = await r.json();
        else throw new Error(`${r.status}: ${await r.text()}`);
      } catch (e) {
        console.error("Eval error:", e);
      }

      setMessages((m) =>
        m.map((msg) =>
          msg.id === messageId ? { ...msg, evaluating: false, evaluation: result } : msg,
        ),
      );
    },
    [backendUrl],
  );

  const phase1Badge = (() => {
    switch (phase1Status) {
      case "complete":
        return { text: "Profile Ready", cls: "text-[var(--accent)] bg-[color-mix(in_oklab,var(--accent)_10%,transparent)] ring-[color-mix(in_oklab,var(--accent)_30%,transparent)]" };
      case "running":
        return { text: "Ingesting", cls: "text-[var(--warning)] bg-[color-mix(in_oklab,var(--warning)_10%,transparent)] ring-[color-mix(in_oklab,var(--warning)_30%,transparent)]" };
      case "failed":
        return { text: "Failed", cls: "text-destructive bg-destructive/5 ring-destructive/20" };
      default:
        return { text: "Awaiting Filing", cls: "text-muted-foreground bg-muted ring-border" };
    }
  })();

  const handleFiles = (files: FileList | null) => {
    if (!files || !files[0]) return;
    setFile(files[0]);
  };

  const phase1Done = phase1Status === "complete";

  const firstIncompleteIdx = steps.findIndex((s) => s.status !== "done");
  const resumeFromStep = firstIncompleteIdx + 1;
  const canResume =
    phase1Status === "failed" &&
    !!savedJobId &&
    resumeFromStep > 0;

  return (
    <div className="min-h-screen font-sans text-foreground selection:bg-[color-mix(in_oklab,var(--accent)_25%,transparent)]">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-white/10 bg-[var(--accent)]">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <BrandLogo className="h-9 w-9 text-white" />
            <div className="flex flex-col leading-tight">
              <span className="font-sans text-lg italic tracking-tight text-white">
                Performance Analysis
              </span>
              <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-white/60">
                KYC Intelligence
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1.5">
              <div
                className={`h-1.5 w-1.5 rounded-full ${
                  connected === null
                    ? "bg-white/30"
                    : connected
                    ? "bg-white"
                    : "bg-red-300"
                }`}
              />
              <span className="font-mono text-[10px] uppercase tracking-widest text-white/70">
                {connected === null ? "Checking" : connected ? "Engine online" : "Engine offline"}
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setColorMode((mode) => mode === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${colorMode === "dark" ? "light" : "dark"} mode`}
              className="text-white/80 hover:text-white hover:bg-white/10"
            >
              {colorMode === "dark" ? <Sun /> : <Moon />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowSettings((s) => !s)}
              aria-label="Settings"
              className="text-white/80 hover:text-white hover:bg-white/10"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
        {showSettings && (
          <div className="border-t border-white/10 bg-card">
            <div className="mx-auto flex max-w-5xl items-center justify-end gap-4 px-6 py-4">
              <div className="flex items-center gap-2" aria-label="Color palette">
                <Palette className="h-4 w-4 text-muted-foreground" />
                {PALETTES.map((item) => (
                  <Button key={item.value} variant="ghost" size="icon" onClick={() => setPalette(item.value)} aria-label={`${item.label} palette`} title={item.label} className={`h-8 w-8 rounded-full ${palette === item.value ? "ring-2 ring-ring ring-offset-2 ring-offset-background" : ""}`}>
                    <span className={`h-4 w-4 rounded-full ${item.previewClass}`} />
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}
      </nav>

      <main className="mx-auto max-w-5xl px-6 pt-12 pb-24">
        {/* Hero */}
        <header className="mb-14 max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 mb-5">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)]" />
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              Counterparty Due Diligence
            </span>
          </div>
          <h1 className="font-sans text-5xl sm:text-6xl leading-[1.02] tracking-tight text-foreground">
            Know who you're <em className="text-[var(--accent)]">really</em> dealing with.
          </h1>
          <p className="mt-5 text-base text-muted-foreground max-w-xl leading-relaxed">
            Turn complex financial reports into a searchable knowledge base. Ask questions in plain language and uncover insights, trends, and risks with source-backed answers.
          </p>
        </header>

        {/* PHASE 1 */}
        <section style={{ animation: "slideUp 0.6s var(--ease-out-expo) both" }}>
          <PhaseHeader
            kicker="01 — Ingestion"
            title="Counterparty Filing"
            subtitle="Drop a 10-K, annual report, or registration document. The system organizes its key financial information, entities, and relationships into an interactive knowledge base."
            badge={phase1Badge}
          />

          <div className="grid gap-6">
            {/* Mode toggle */}
            <div className="flex rounded-xl border border-border bg-card p-1 gap-1 shadow-[var(--shadow-soft)]">
              <button
                onClick={() => setIngestionMode("upload")}
                disabled={phase1Status === "running"}
                className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition-all disabled:cursor-not-allowed ${
                  ingestionMode === "upload"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Upload className="h-3.5 w-3.5" /> Upload New Filing
              </button>
              <button
                onClick={() => { setIngestionMode("select"); if (companies.length === 0) void loadCompanies(); }}
                disabled={phase1Status === "running"}
                className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                  ingestionMode === "select"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Database className="h-3.5 w-3.5" />
                Use Existing Company
                <span className={`font-mono text-[10px] rounded-full px-1.5 py-0.5 ${ingestionMode === "select" ? "bg-white/20 text-white" : "bg-muted text-muted-foreground"}`}>
                  {companiesLoading ? "…" : companies.length}
                </span>
              </button>
            </div>

            {/* Upload card */}
            {ingestionMode === "upload" && (
            <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-[var(--shadow-soft)]">
              <div className="border-b border-border bg-[var(--panel)] px-5 py-3 flex items-center justify-between">
                <div className="flex items-center gap-2 text-foreground">
                  <Building2 className="h-3.5 w-3.5" />
                  <span className="font-mono text-[10px] uppercase tracking-widest">Subject Entity</span>
                </div>
                <span className="font-mono text-[10px] text-muted-foreground">.html · .htm · .pdf</span>
              </div>

              <div className="p-5">
                <label
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
                  className={`relative flex h-36 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-all ${
                    dragOver
                      ? "border-[var(--accent)] bg-[color-mix(in_oklab,var(--accent)_5%,transparent)]"
                      : "border-border hover:border-[var(--accent)]/50"
                  }`}
                >
                  <input ref={fileInputRef} type="file" accept=".html,.htm,.pdf" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[var(--elevated)] text-[var(--accent)]">
                    <Upload className="h-4 w-4" />
                  </div>
                  <span className="text-sm font-medium text-foreground">
                    Drop a counterparty filing or <span className="text-[var(--accent)] underline-offset-4 hover:underline">browse files</span>
                  </span>
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    SEC 10-K · Annual Report · Registration Doc
                  </span>
                </label>

                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted shrink-0">
                      <FileText className="h-3.5 w-3.5 text-muted-foreground" />
                    </div>
                    <span className="text-sm font-medium truncate">
                      {file ? file.name : previousFilename && phase1Status === "complete" ? previousFilename : "No file selected"}
                    </span>
                    {file && <span className="font-mono text-[10px] text-muted-foreground uppercase shrink-0">{fmtBytes(file.size)}</span>}
                    {!file && previousFilename && phase1Status === "complete" && (
                      <span className="font-mono text-[10px] text-[var(--accent)] uppercase shrink-0">restored</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-2 rounded-md border border-input bg-background px-2.5 py-1.5">
                      <label className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">FY</label>
                      <input type="text" value={fiscalYear} onChange={(e) => setFiscalYear(e.target.value)} className="w-14 bg-transparent text-xs font-mono outline-none text-foreground" />
                    </div>
                    <button
                      onClick={canResume ? resumePipeline : runPipeline}
                      disabled={(canResume ? false : !file) || phase1Status === "running"}
                      className="group relative flex items-center gap-2 overflow-hidden rounded-md bg-[var(--elevated)] text-[var(--accent)] text-xs font-semibold px-4 py-2 transition-all hover:bg-[var(--elevated)] disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                    >
                      {canResume ? (
                        <><RotateCcw className="h-3.5 w-3.5" /> Resume from step {resumeFromStep}</>
                      ) : phase1Status === "complete" || phase1Status === "failed" ? (
                        <><RotateCcw className="h-3.5 w-3.5" /> Re-ingest</>
                      ) : (
                        <><Play className="h-3.5 w-3.5 fill-current" /> Run Diligence</>
                      )}
                    </button>
                  </div>
                </div>

                {phase1Error && (
                  <div className="mt-3 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-2.5 text-xs text-destructive">
                    <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                    <span className="font-mono">{phase1Error}</span>
                  </div>
                )}
              </div>
            </div>
            )}

            {/* Select existing company */}
            {ingestionMode === "select" && (
              <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-[var(--shadow-soft)]" style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
                <div className="border-b border-border bg-[var(--panel)] px-5 py-3 flex items-center gap-2">
                  <Database className="h-3.5 w-3.5 text-[var(--accent)]" />
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Knowledge Graph</span>
                  <span className="ml-auto font-mono text-[10px] text-[var(--accent)]">{companies.length} {companies.length === 1 ? "company" : "companies"}</span>
                </div>
                <div className="p-6 flex flex-col gap-5">
                  <div>
                    <p className="text-sm font-medium text-foreground">Select target company</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Choose a company already stored in the knowledge graph. All answers and peer comparisons will be scoped to this target.
                    </p>
                  </div>
                  {companiesLoading ? (
                    <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2.5 text-sm text-muted-foreground">
                      <span className="h-3 w-3 rounded-full border-2 border-[var(--accent)] border-t-transparent animate-spin" />
                      Loading companies from graph…
                    </div>
                  ) : companies.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-background px-4 py-5 text-center">
                      <Database className="h-5 w-5 mx-auto text-muted-foreground/40 mb-2" />
                      <p className="text-xs text-muted-foreground">No target companies found in the graph.</p>
                      <p className="text-xs text-muted-foreground/60 mt-0.5">Upload a filing first to populate the knowledge graph.</p>
                    </div>
                  ) : (
                    <select
                      value={selectedCompany}
                      onChange={(e) => setSelectedCompany(e.target.value)}
                      className="w-full rounded-lg border border-input bg-background px-3 py-2.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
                      aria-label="Select target company"
                    >
                      {companies.map((c) => (
                        <option key={c.name} value={c.name}>{c.name}</option>
                      ))}
                    </select>
                  )}
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => void loadCompanies()}
                      disabled={companiesLoading}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
                    >
                      ↻ Refresh
                    </button>
                    <button
                      onClick={() => {
                        setPhase1Status("complete");
                        try { localStorage.setItem("verdant_phase1_status", "complete"); } catch { /* ignore */ }
                        setTimeout(() => phase2Ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
                      }}
                      disabled={!selectedCompany || companies.length === 0}
                      className="flex items-center gap-2 rounded-md bg-[var(--accent)] text-white text-xs font-semibold px-5 py-2.5 transition-all hover:brightness-105 disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                    >
                      <Play className="h-3.5 w-3.5 fill-current" /> Start Analysis
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Stepper — only relevant during / after upload pipeline */}
            {ingestionMode === "upload" && (phase1Status !== "idle" || steps.some((s) => s.status !== "pending")) && (
              <div
                className="rounded-2xl border border-border bg-card p-6 shadow-[var(--shadow-soft)]"
                style={{ animation: "fadeReveal 0.5s var(--ease-out-expo) both" }}
              >
                <div className="relative space-y-0 pl-2">
                  <div className="absolute left-4 top-2 bottom-2 w-px bg-border" />
                  {steps.map((s, i) => (
                    <StepItem key={s.id} step={s} isLast={i === steps.length - 1} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Summary panel — visible after pipeline completes */}
        {phase1Done && (
          <div
            className="mt-8 rounded-2xl border border-border bg-card overflow-hidden shadow-[var(--shadow-soft)]"
            style={{ animation: "fadeReveal 0.6s var(--ease-out-expo) both" }}
          >
            <div className="border-b border-border bg-[var(--panel)] px-5 py-3 flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                Pipeline Summary
              </span>
            </div>
            <div className="divide-y divide-border">
              {steps.filter((s) => s.summary).map((s) => (
                <div key={s.id} className="flex items-start gap-4 px-5 py-3">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--accent)] w-4 shrink-0 mt-0.5">
                    {String(s.id).padStart(2, "0")}
                  </span>
                  <span className="font-mono text-[10px] w-36 shrink-0 text-muted-foreground truncate">
                    {s.label}
                  </span>
                  <span className="text-xs text-foreground leading-relaxed">{s.summary}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* PHASE 2 — appears smoothly after Phase 1 complete */}
        {phase1Done && (
          <section
            ref={phase2Ref}
            className="mt-20"
            style={{ animation: "fadeReveal 0.8s var(--ease-out-expo) both" }}
          >
            <PhaseHeader
              kicker="02 — Inquiry"
              title="Ask about the counterparty"
              subtitle="Query the knowledge graph in natural language. Every answer ships with a process trace and can be audited on demand."
              badge={{
                text: "Ready",
                cls: "text-[var(--accent)] bg-[color-mix(in_oklab,var(--accent)_10%,transparent)] ring-[color-mix(in_oklab,var(--accent)_30%,transparent)]",
              }}
            />

            <div className="mb-4 flex flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-[var(--shadow-soft)] sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-semibold text-foreground">Company in focus</p>
                <p className="mt-0.5 text-xs text-muted-foreground">Answers and comparisons are scoped to this target company.</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 rounded-lg border border-border bg-[var(--elevated)] px-3 py-2 sm:min-w-56">
                  <Building2 className="h-4 w-4 shrink-0 text-[var(--accent)]" />
                  <span className="text-sm font-medium text-foreground truncate">{selectedCompany || "—"}</span>
                </div>
                <button
                  onClick={async () => {
                    if (showGraph && graphData) { setShowGraph(false); return; }
                    setGraphLoading(true);
                    try {
                      const r = await fetch(`${backendUrl}/graph/explore?company=${encodeURIComponent(selectedCompany)}`);
                      if (r.ok) { setGraphData(await r.json()); setShowGraph(true); }
                    } finally { setGraphLoading(false); }
                  }}
                  disabled={!selectedCompany || graphLoading}
                  className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground hover:border-[var(--accent)] transition-all disabled:opacity-40"
                >
                  {graphLoading
                    ? <span className="h-3 w-3 rounded-full border-2 border-[var(--accent)] border-t-transparent animate-spin" />
                    : <Database className="h-3.5 w-3.5" />}
                  Explore graph
                </button>
              </div>
            </div>

            {showGraph && graphData && (
              <GraphExplorer data={graphData} onClose={() => setShowGraph(false)} />
            )}

            <div className="rounded-2xl border border-border bg-card shadow-[var(--shadow-soft)] overflow-hidden">
              {/* Messages */}
              <div className="divide-y divide-border max-h-[520px] overflow-y-auto">
                {messages.length === 0 && (
                  <div className="p-10 text-center">
                    <Sparkles className="h-5 w-5 mx-auto text-[var(--accent)] mb-3" />
                    <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                      Try: <span className="text-foreground italic">"What are this company's top concentration risks compared to its peers?"</span>
                    </p>
                  </div>
                )}
                {messages.map((m) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    onEvaluate={(testType) => runEvaluation(m.id, testType)}
                    sourceFileName={file?.name ?? previousFilename ?? null}
                  />
                ))}
              </div>

              {/* Composer */}
              <div className="border-t border-border bg-[var(--panel)] p-3 flex items-center gap-2">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askQuestion(); } }}
                  placeholder="Ask about risks, financials, peer comparisons…"
                  disabled={asking}
                  className="flex-1 rounded-lg bg-card border border-input px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-ring transition-all placeholder:text-muted-foreground"
                />
                <button
                  type="button"
                  onClick={() => setWithReasoning((r) => !r)}
                  title={withReasoning ? "Reasoning on — click to disable" : "Reasoning off — click to enable"}
                  className={`flex items-center gap-1 rounded-lg border px-3 py-2.5 text-xs font-semibold transition-all ${
                    withReasoning
                      ? "border-[var(--accent)] bg-[color-mix(in_oklab,var(--accent)_12%,transparent)] text-[var(--accent)]"
                      : "border-border bg-card text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={askQuestion}
                  disabled={!question.trim() || asking}
                  className="flex items-center gap-1.5 rounded-lg bg-[var(--warning)] text-white text-xs font-semibold px-4 py-2.5 transition-all hover:brightness-105 disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                >
                  <Send className="h-3.5 w-3.5" /> Ask
                </button>
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-24 pt-8 border-t border-border flex items-center gap-2 text-muted-foreground">
          <ShieldCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
          <span className="text-[10px] font-mono uppercase tracking-widest">
            KYC Engine · Auditable by design
          </span>
        </footer>
      </main>
    </div>
  );
}

function PhaseHeader({
  kicker,
  title,
  subtitle,
  badge,
}: {
  kicker: string;
  title: string;
  subtitle: string;
  badge: { text: string; cls: string };
}) {
  return (
    <div className="mb-8 flex items-start justify-between gap-4">
      <div className="max-w-xl">
        <div className="font-mono text-[10px] font-bold tracking-[0.2em] text-[var(--warning)] uppercase mb-2">
          {kicker}
        </div>
        <h2 className="font-sans text-3xl tracking-tight text-foreground">{title}</h2>
        <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{subtitle}</p>
      </div>
      <span
        className={`mt-1 text-[10px] font-mono uppercase tracking-widest px-2.5 py-1 rounded-full ring-1 whitespace-nowrap ${badge.cls}`}
      >
        {badge.text}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CitedText — renders answer text with [CITE:id] as clickable superscripts
// ---------------------------------------------------------------------------

const _CITE_RE = /\[CITE:([^\]]+)\]/g;

function CitedText({
  text,
  citations,
  sourceFileName,
}: {
  text: string;
  citations?: Record<string, CitationInfo>;
  sourceFileName?: string | null;
}) {
  // Build ordered list of unique citation IDs as they appear in the text
  const citeOrder: string[] = [];
  const seen = new Set<string>();
  for (const m of text.matchAll(_CITE_RE)) {
    if (!seen.has(m[1])) { seen.add(m[1]); citeOrder.push(m[1]); }
  }

  // Render the Sources section as links too (lines starting with "- [")
  const renderLine = (line: string, lineIdx: number) => {
    // Markdown link: [label](url)
    const mdLink = /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g;
    const linkParts: React.ReactNode[] = [];
    let pos = 0;
    for (const m of line.matchAll(new RegExp(mdLink.source, "g"))) {
      if (m.index! > pos) linkParts.push(line.slice(pos, m.index));
      linkParts.push(
        <a key={m.index} href={m[2]} target="_blank" rel="noopener noreferrer"
           className="text-[var(--accent)] underline underline-offset-2 hover:opacity-80 break-all">
          {m[1]}
        </a>
      );
      pos = m.index! + m[0].length;
    }
    if (pos < line.length) linkParts.push(line.slice(pos));
    return <span key={lineIdx}>{linkParts}</span>;
  };

  // Split answer into lines to handle Sources section markdown
  const lines = text.split("\n");
  const lineNodes = lines.map((rawLine, li) => {
    const isBold = rawLine.startsWith("**") || rawLine.startsWith("## ");
    const isListItem = rawLine.startsWith("- ") || rawLine.startsWith("* ");
    // Process [CITE:] within a line
    const lineSegments: React.ReactNode[] = [];
    let lpos = 0;
    for (const m of rawLine.matchAll(new RegExp(_CITE_RE.source, "g"))) {
      if (m.index! > lpos) lineSegments.push(renderLine(rawLine.slice(lpos, m.index!), lpos));
      const n = citeOrder.indexOf(m[1]) + 1;
      const info = citations?.[m[1]];
      const href = info?.document_url;
      lineSegments.push(
        href ? (
          <a key={m.index} href={href} target="_blank" rel="noopener noreferrer"
             title={info.summary || m[1]}
             className="ml-0.5 align-super text-[9px] font-mono text-[var(--accent)] hover:underline">
            [{n}]
          </a>
        ) : (
          <sup key={m.index} title={info?.summary || m[1]}
               className="ml-0.5 text-[9px] font-mono text-[var(--accent)]/70">
            [{n}]
          </sup>
        )
      );
      lpos = m.index! + m[0].length;
    }
    if (lpos < rawLine.length) lineSegments.push(renderLine(rawLine.slice(lpos), lpos));

    const inner = lineSegments.length > 0 ? lineSegments : renderLine(rawLine, li);

    if (isBold) return <p key={li} className="font-semibold text-foreground mt-2">{inner}</p>;
    if (isListItem) return <li key={li} className="ml-4 list-disc text-muted-foreground">{inner}</li>;
    return rawLine.trim() === ""
      ? <div key={li} className="h-2" />
      : <p key={li} className="text-sm leading-relaxed text-foreground">{inner}</p>;
  });

  // Footnote list at bottom
  const footnotes = citeOrder
    .map((id, idx) => {
      const info = citations?.[id];
      if (!info) return null;
      const href = info.document_url;
      const page = info.source_page != null && String(info.source_page).trim() !== "" ? String(info.source_page) : null;
      const section = info.section_title?.trim() || null;
      const isTarget = info.role === "target";
      const showFileFallback = !href && isTarget && !!sourceFileName;

      return (
        <div key={id} className="flex gap-2 text-[10px] font-mono text-muted-foreground">
          <span className="shrink-0 text-[var(--accent)]">[{idx + 1}]</span>
          <span className="min-w-0">
            <span className="text-foreground">{info.company}</span>
            {" · "}
            <span className={isTarget ? "text-[var(--accent)]" : "text-[var(--warning)]"}>
              {info.role}
            </span>
            {" · "}
            {info.summary.slice(0, 80)}
            {section && <span className="text-muted-foreground"> · {section}</span>}
            {showFileFallback ? (
              <>
                {" · "}
                <span className="text-foreground">Source: {sourceFileName}</span>
                {page && <span className="text-muted-foreground"> · p. {page}</span>}
              </>
            ) : href ? (
              <>
                {" "}
                <a href={href} target="_blank" rel="noopener noreferrer"
                   className="text-[var(--accent)] underline underline-offset-2 hover:opacity-80">
                  source ↗
                </a>
                {page && <span className="text-muted-foreground"> · p. {page}</span>}
              </>
            ) : (
              page && <span className="text-muted-foreground"> · p. {page}</span>
            )}
          </span>
        </div>
      );
    })
    .filter(Boolean);

  return (
    <div className="space-y-0.5">
      <div className="space-y-1">{lineNodes}</div>
      {footnotes.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border space-y-1.5">
          <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
            Citations
          </span>
          {footnotes}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ReasoningTrace — Claude-style muted "thought" block shown UNDER the answer
// ---------------------------------------------------------------------------

function ReasoningTrace({ trace }: { trace: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mt-4 rounded-lg border border-border/60 bg-[var(--panel)]/60 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-muted/40 transition-colors"
      >
        <span className="flex items-center gap-2 text-[11px] italic text-muted-foreground">
          <Sparkles className="h-3 w-3 opacity-60" />
          Reasoning
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 text-muted-foreground/70 transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>
      {open && (
        <div className="px-4 pb-3 pt-1 border-t border-border/40">
          <pre className="whitespace-pre-wrap text-[12px] font-sans text-muted-foreground/80 leading-relaxed max-h-96 overflow-y-auto italic">
            {trace}
          </pre>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MessageBubble
// ---------------------------------------------------------------------------

function MessageBubble({
  message,
  onEvaluate,
  sourceFileName,
}: {
  message: ChatMessage;
  onEvaluate: (testType: string) => void;
  sourceFileName?: string | null;
}) {
  const [traceOpen, setTraceOpen] = useState(false);
  const [selectedEval, setSelectedEval] = useState("overall_score");

  if (message.role === "user") {
    return (
      <div className="p-5 flex justify-end" style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-[var(--elevated)] text-[var(--accent)] px-4 py-2.5 text-sm font-medium shadow-[var(--shadow-soft)]">
          {message.content}
        </div>
      </div>
    );
  }

  const isThinking = !message.content;
  const traceDone = message.trace?.every((t) => t.status === "done");

  // Auto-run an overall score once the answer is ready, so the user always
  // sees a score under the answer. They can still switch tests afterwards.
  const autoEvalFiredRef = useRef(false);
  useEffect(() => {
    if (
      !isThinking &&
      traceDone &&
      !message.evaluation &&
      !message.evaluating &&
      !autoEvalFiredRef.current
    ) {
      autoEvalFiredRef.current = true;
      onEvaluate("overall_score");
    }
  }, [isThinking, traceDone, message.evaluation, message.evaluating, onEvaluate]);

  return (
    <div className="p-5" style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
      <div className="flex gap-3">
        <div className="h-7 w-7 shrink-0 rounded-full bg-[var(--accent)] flex items-center justify-center text-white shadow-[var(--shadow-soft)]">
          <Sparkles className="h-3.5 w-3.5" />
        </div>
        <div className="flex-1 min-w-0">
          {/* Process trace */}
          {message.trace && (
            <div className="mb-3 rounded-lg border border-border bg-[var(--panel)] overflow-hidden">
              <button
                onClick={() => setTraceOpen((o) => !o)}
                className="w-full flex items-center justify-between px-3 py-2 text-left"
              >
                <span className="font-mono text-[10px] uppercase tracking-widest text-foreground">
                  {isThinking ? "Reasoning…" : "Process trace"}
                </span>
                <div className="flex items-center gap-2">
                  {isThinking && (
                    <span className="font-mono text-[10px] text-[var(--warning)] animate-pulse">live</span>
                  )}
                  <ChevronDown
                    className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${
                      traceOpen || isThinking ? "rotate-180" : ""
                    }`}
                  />
                </div>
              </button>
              {(traceOpen || isThinking) && (
                <div className="px-3 pb-3 space-y-1.5 font-mono text-[10px]">
                  {message.trace.map((t) => (
                    <div key={t.step} className="flex items-center justify-between">
                      <span className="flex items-center gap-2 text-muted-foreground">
                        <StatusDot status={t.status} />
                        {t.step}
                      </span>
                      <span
                        className={
                          t.status === "done"
                            ? "text-[var(--accent)]"
                            : t.status === "running"
                            ? "text-[var(--warning)]"
                            : "text-muted-foreground/60"
                        }
                      >
                        {t.status.toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Answer */}
          {isThinking ? (
            <div className="h-4 w-2/3 rounded shimmer-bg" />
          ) : (
            <CitedText
              text={message.content}
              citations={message.citations}
              sourceFileName={sourceFileName}
            />
          )}

          {/* Reasoning trace (LLM chain-of-thought) — Claude-style muted block */}
          {message.reasoning_trace && (
            <ReasoningTrace trace={message.reasoning_trace} />
          )}

          {/* Evaluation panel — always visible after the answer, with test switcher */}
          {!isThinking && traceDone && (
            <div className="mt-5 rounded-xl border border-border bg-[var(--panel)]/50 overflow-hidden">
              <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-border/60">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
                  <span className="font-mono text-[10px] uppercase tracking-widest text-foreground">
                    Fidelity Audit
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={selectedEval}
                    onChange={(e) => setSelectedEval(e.target.value)}
                    disabled={message.evaluating}
                    className="rounded-md border border-input bg-background px-2 py-1 text-[11px] font-mono outline-none focus:ring-1 focus:ring-ring text-foreground disabled:opacity-50"
                  >
                    {EVAL_TESTS.map((t) => (
                      <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => onEvaluate(selectedEval)}
                    disabled={message.evaluating}
                    className="inline-flex items-center gap-1.5 rounded-md bg-[var(--accent)] text-white px-3 py-1 text-[11px] font-semibold hover:brightness-110 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                  >
                    {message.evaluation && message.evalType === selectedEval ? "Re-run" : "Run"}
                  </button>
                </div>
              </div>

              {message.evaluating && (
                <div className="flex items-center gap-2 px-4 py-4 text-[11px] text-muted-foreground font-mono uppercase tracking-widest">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)] animate-pulse" />
                  Running {EVAL_TESTS.find((t) => t.value === message.evalType)?.label ?? "audit"}…
                </div>
              )}

              {!message.evaluating && !message.evaluation && (
                <div className="px-4 py-4 text-[11px] text-muted-foreground italic">
                  Pick a test above and run it to score this answer.
                </div>
              )}

              {message.evaluation && !message.evaluating && (
                <Scorecard data={message.evaluation} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  if (status === "done")
    return <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />;
  if (status === "running")
    return (
      <span
        className="h-1.5 w-1.5 rounded-full bg-[var(--warning)]"
        style={{ animation: "pulseSlow 1.2s infinite" }}
      />
    );
  return <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30" />;
}

function Scorecard({ data }: { data: EvaluationResult }) {
  const label = EVAL_TESTS.find((t) => t.value === data.test_type)?.label ?? "Audit";
  return (
    <div
      className="mt-2 overflow-hidden rounded-xl border border-border bg-card"
      style={{ animation: "fadeReveal 0.5s var(--ease-out-expo) both" }}
    >
      <div className="flex items-center justify-between px-4 py-3 bg-[var(--panel)] border-b border-border">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-foreground">
            {label}
          </span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Score
          </span>
          <span className="font-sans text-2xl text-[var(--accent)]">
            {(data.weighted * 100).toFixed(0)}
          </span>
          <span className="text-xs text-muted-foreground">/100</span>
        </div>
      </div>
      <table className="w-full text-left">
        <tbody className="divide-y divide-border text-sm">
          {data.rows.map((r, i) => (
            <tr key={i}>
              <td className="px-4 py-2.5 font-medium text-foreground w-1/3 truncate max-w-[160px]" title={r.dimension}>
                {r.dimension}
              </td>
              <td className="px-4 py-2.5 text-xs text-muted-foreground">{r.note}</td>
              <td className="px-4 py-2.5 w-32">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[var(--accent)]"
                      style={{ width: `${r.score * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-foreground w-8 text-right">
                    {(r.score * 100).toFixed(0)}
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// GraphExplorer — Canvas-based knowledge graph viewer
// ---------------------------------------------------------------------------

interface GNode { id: string; name: string; type: string; group: string; [key: string]: unknown; }
interface GLink { source: string; target: string; type: string; }
interface GraphData { nodes: GNode[]; links: GLink[]; }

const GROUP_COLOR: Record<string, string> = {
  target:          "#7d141c",
  peer:            "#fbf5f7",
  industry:        "#9f4780",
  metric:          "#24454d",
  metric_category: "#24454d",
  metric_value:    "#f9e091",
  risk:            "#e3726c",
  peer_risk:       "#e3726c",
  siccode:         "#959aa1",
};

// Base unit = 10. target = ×4, peer = ×2, everything else = ×1
const BASE_RADIUS = 10;
const GROUP_RADIUS: Record<string, number> = {
  target:          BASE_RADIUS * 4,
  peer:            BASE_RADIUS * 2,
  industry:        BASE_RADIUS,
  metric:          BASE_RADIUS,
  metric_category: BASE_RADIUS,
  metric_value:    BASE_RADIUS,
  risk:            BASE_RADIUS,
  peer_risk:       BASE_RADIUS,
  siccode:         BASE_RADIUS,
};

const SKIP_PROPS = new Set(["id", "name", "type", "group"]);

function GraphExplorer({ data, onClose }: { data: GraphData; onClose: () => void }) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const posRef       = useRef<Map<string, { x: number; y: number }>>(new Map());
  const velRef       = useRef<Map<string, { vx: number; vy: number }>>(new Map());
  const dragRef      = useRef<{ id: string | null; ox: number; oy: number; moved: boolean }>({ id: null, ox: 0, oy: 0, moved: false });
  const panRef       = useRef<{ dragging: boolean; sx: number; sy: number; ox: number; oy: number }>({ dragging: false, sx: 0, sy: 0, ox: 0, oy: 0 });
  const offsetRef    = useRef({ x: 0, y: 0 });
  const scaleRef     = useRef(1);
  const selectedRef  = useRef<string | null>(null);
  const rafRef       = useRef<number>(0);
  const [tooltip, setTooltip]       = useState<{ x: number; y: number; text: string } | null>(null);
  const [selectedNode, setSelectedNode] = useState<GNode | null>(null);
  const [cursor, setCursor]         = useState<string>("grab");

  // Map from node id to neighbouring node ids (for the detail panel)
  const neighborMap = useRef<Map<string, string[]>>(new Map());
  useEffect(() => {
    const m = new Map<string, string[]>();
    data.nodes.forEach((n) => m.set(n.id, []));
    data.links.forEach((l) => {
      m.get(l.source)?.push(l.target);
      m.get(l.target)?.push(l.source);
    });
    neighborMap.current = m;
  }, [data]);

  // Translate a CSS-pixel canvas event position to canvas pixel coords
  const toCv = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width  / rect.width),
      y: (e.clientY - rect.top)  * (canvas.height / rect.height),
    };
  }, []);

  // World ↔ screen transform
  const toScreen = (wx: number, wy: number) => ({
    x: wx * scaleRef.current + offsetRef.current.x,
    y: wy * scaleRef.current + offsetRef.current.y,
  });
  const toWorld = (sx: number, sy: number) => ({
    x: (sx - offsetRef.current.x) / scaleRef.current,
    y: (sy - offsetRef.current.y) / scaleRef.current,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.nodes.length) return;
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;

    // Centre the view initially
    offsetRef.current = { x: cx, y: cy };

    // Radial initial positions (world space, centred at 0,0)
    const groups = ["target", "peer", "metric", "industry", "risk", "peer_risk", "metric_category", "metric_value", "siccode"];
    const radii  = [0, 160, 280, 280, 380, 380, 380, 480, 280];
    const byGroup: Record<string, GNode[]> = {};
    data.nodes.forEach((n) => { (byGroup[n.group] = byGroup[n.group] || []).push(n); });

    data.nodes.forEach((n) => {
      const gIdx  = groups.indexOf(n.group);
      const r     = gIdx >= 0 ? radii[gIdx] : 250;
      const peers = byGroup[n.group] || [];
      const idx   = peers.indexOf(n);
      const total = peers.length;
      const angle = total > 1 ? (idx / total) * 2 * Math.PI - Math.PI / 2 : 0;
      posRef.current.set(n.id, { x: r * Math.cos(angle), y: r * Math.sin(angle) });
      velRef.current.set(n.id, { x: 0, y: 0 } as any);
    });

    const resolveColor = (g: string) => GROUP_COLOR[g] || "#888";

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const scale = scaleRef.current;
      const { x: ox, y: oy } = offsetRef.current;
      ctx.clearRect(0, 0, W, H);

      // links
      data.links.forEach((l) => {
        const s = posRef.current.get(l.source), t = posRef.current.get(l.target);
        if (!s || !t) return;
        const ss = toScreen(s.x, s.y), ts = toScreen(t.x, t.y);
        ctx.beginPath();
        ctx.moveTo(ss.x, ss.y);
        ctx.lineTo(ts.x, ts.y);

        // highlight edges connected to selected node
        const selId = selectedRef.current;
        const isHighlight = selId && (l.source === selId || l.target === selId);
        ctx.strokeStyle = isHighlight ? "rgba(180,180,255,0.6)" : "rgba(128,128,128,0.22)";
        ctx.lineWidth   = isHighlight ? 2 : 1;
        ctx.stroke();

        // edge label
        if (scale > 0.7 && l.type) {
          const mx = (ss.x + ts.x) / 2, my = (ss.y + ts.y) / 2;
          ctx.save();
          ctx.font = `${Math.round(9 * scale)}px sans-serif`;
          ctx.fillStyle = "rgba(150,150,150,0.8)";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(l.type, mx, my);
          ctx.restore();
        }
      });

      // nodes
      data.nodes.forEach((n) => {
        const p = posRef.current.get(n.id);
        if (!p) return;
        const { x: px, y: py } = toScreen(p.x, p.y);
        const baseR = GROUP_RADIUS[n.group] ?? 10;
        const r = baseR * scale;
        const isSelected = selectedRef.current === n.id;
        const color = resolveColor(n.group);

        if (isSelected) {
          ctx.beginPath();
          ctx.arc(px, py, r + 5, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(255,255,255,0.18)";
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        if (isSelected) {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }

        if (scale > 0.45) {
          ctx.fillStyle = "#fff";
          ctx.font = `${Math.round((n.group === "target" ? 11 : 9) * Math.min(scale, 1.4))}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          const label = n.name.length > 20 ? n.name.slice(0, 19) + "…" : n.name;
          ctx.fillText(label, px, py + r + Math.round(11 * Math.min(scale, 1.2)));
        }
      });
    };

    // Simple force sim
    const tick = () => {
      const alpha   = 0.05;
      const repulse = 1800;
      const attract = 0.04;
      const ids     = data.nodes.map((n) => n.id);

      ids.forEach((id) => {
        if (dragRef.current.id === id) return; // don't simulate dragged node
        const p = posRef.current.get(id)!;
        const v = velRef.current.get(id) as any;

        ids.forEach((other) => {
          if (other === id) return;
          const o = posRef.current.get(other)!;
          const dx = p.x - o.x, dy = p.y - o.y;
          const d2 = dx * dx + dy * dy + 1;
          const f = repulse / d2;
          v.x += f * dx; v.y += f * dy;
        });

        data.links.forEach((l) => {
          let other: string | null = null;
          if (l.source === id) other = l.target;
          if (l.target === id) other = l.source;
          if (!other) return;
          const o = posRef.current.get(other)!;
          const dx = o.x - p.x, dy = o.y - p.y;
          v.x += dx * attract; v.y += dy * attract;
        });

        // gentle gravity toward world origin
        v.x += (0 - p.x) * 0.003;
        v.y += (0 - p.y) * 0.003;

        v.x *= 0.85; v.y *= 0.85;
        p.x += v.x * alpha; p.y += v.y * alpha;
      });

      draw();
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const hitTest = useCallback((cvx: number, cvy: number) => {
    const { x: wx, y: wy } = toWorld(cvx, cvy);
    for (const n of [...data.nodes].reverse()) {
      const p = posRef.current.get(n.id);
      if (!p) continue;
      const r = GROUP_RADIUS[n.group] ?? 10;
      const dx = wx - p.x, dy = wy - p.y;
      if (dx * dx + dy * dy <= r * r) return n;
    }
    return null;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const onWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current!;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const cvx = (e.clientX - rect.left) * (canvas.width  / rect.width);
    const cvy = (e.clientY - rect.top)  * (canvas.height / rect.height);
    const factor   = e.deltaY < 0 ? 1.12 : 0.89;
    const newScale = Math.max(0.08, Math.min(6, scaleRef.current * factor));
    const ratio    = newScale / scaleRef.current;
    offsetRef.current = {
      x: cvx - (cvx - offsetRef.current.x) * ratio,
      y: cvy - (cvy - offsetRef.current.y) * ratio,
    };
    scaleRef.current = newScale;
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = toCv(e);
    const hit = hitTest(cv.x, cv.y);
    if (hit) {
      const wp = posRef.current.get(hit.id)!;
      dragRef.current = { id: hit.id, ox: cv.x - toScreen(wp.x, wp.y).x, oy: cv.y - toScreen(wp.x, wp.y).y, moved: false };
      setCursor("grabbing");
    } else {
      panRef.current = { dragging: true, sx: e.clientX, sy: e.clientY, ox: offsetRef.current.x, oy: offsetRef.current.y };
      setCursor("grabbing");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hitTest, toCv]);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = toCv(e);
    if (dragRef.current.id) {
      dragRef.current.moved = true;
      const wp = toWorld(cv.x - dragRef.current.ox, cv.y - dragRef.current.oy);
      // ox/oy in dragRef are CSS offset, but we need to map directly:
      // screen pos of node = cv - (ox, oy) → world pos
      const p = posRef.current.get(dragRef.current.id);
      if (p) {
        const ws = toWorld(cv.x - dragRef.current.ox, cv.y - dragRef.current.oy);
        p.x = ws.x; p.y = ws.y;
      }
      setCursor("grabbing");
    } else if (panRef.current.dragging) {
      const canvas = canvasRef.current!;
      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      offsetRef.current = {
        x: panRef.current.ox + (e.clientX - panRef.current.sx) * scaleX,
        y: panRef.current.oy + (e.clientY - panRef.current.sy) * scaleY,
      };
      setCursor("grabbing");
    } else {
      const hit = hitTest(cv.x, cv.y);
      setTooltip(hit ? { x: e.clientX, y: e.clientY, text: `${hit.type}: ${hit.name}` } : null);
      setCursor(hit ? "pointer" : "grab");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hitTest, toCv]);

  const onMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current.id && !dragRef.current.moved) {
      // treat as click — select the node
      const n = data.nodes.find((n) => n.id === dragRef.current.id) ?? null;
      selectedRef.current = n?.id ?? null;
      setSelectedNode(n);
    }
    if (!dragRef.current.id && !panRef.current.dragging) {
      // click on empty canvas → deselect
      const cv = toCv(e);
      const hit = hitTest(cv.x, cv.y);
      if (!hit) { selectedRef.current = null; setSelectedNode(null); }
    }
    dragRef.current.id = null;
    dragRef.current.moved = false;
    panRef.current.dragging = false;
    setCursor("grab");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, hitTest, toCv]);

  const onMouseLeave = useCallback(() => {
    dragRef.current.id = null;
    panRef.current.dragging = false;
    setTooltip(null);
    setCursor("grab");
  }, []);

  const onClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = toCv(e);
    const hit = hitTest(cv.x, cv.y);
    selectedRef.current = hit?.id ?? null;
    setSelectedNode(hit ?? null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hitTest, toCv]);

  const GROUP_LABEL: Record<string, string> = {
    target: "Target Company", peer: "Company", industry: "Industry",
    metric: "Metric", metric_category: "Metric Category",
    risk: "Risk", peer_risk: "Peer Risk", siccode: "SIC Code",
  };
  const legend = Object.entries(GROUP_COLOR).map(([g, c]) => ({ group: g, label: GROUP_LABEL[g] ?? g, color: c }));

  // Extra properties of the selected node (beyond id/name/type/group)
  const extraProps = selectedNode
    ? Object.entries(selectedNode).filter(([k]) => !SKIP_PROPS.has(k))
    : [];
  const neighbors = selectedNode
    ? (neighborMap.current.get(selectedNode.id) ?? [])
        .map((nid) => data.nodes.find((n) => n.id === nid))
        .filter(Boolean) as GNode[]
    : [];
  const nodeColor = selectedNode ? (GROUP_COLOR[selectedNode.group] || "#888") : "#888";

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background/95 backdrop-blur-sm">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
        <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Knowledge Graph Explorer</span>
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-3 flex-wrap">
            {legend.map(({ group, label, color }) => (
              <span key={group} className="flex items-center gap-1 text-[10px] font-mono text-muted-foreground">
                <span className="h-2.5 w-2.5 rounded-full inline-block border border-white/10" style={{ background: color }} />
                {label}
              </span>
            ))}
          </div>
          <span className="font-mono text-[10px] text-muted-foreground/60">Scroll to zoom · Drag to pan · Click node for details</span>
          <button onClick={onClose} className="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted transition-colors">
            Close
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Canvas */}
        <div className="relative flex-1 overflow-hidden">
          <canvas
            ref={canvasRef}
            width={1600}
            height={900}
            style={{ width: "100%", height: "100%", cursor }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseLeave}
            onClick={onClick}
            onWheel={onWheel}
          />
          {tooltip && !selectedNode && (
            <div
              className="pointer-events-none fixed rounded-md border border-border bg-card px-2.5 py-1.5 text-xs shadow-lg max-w-xs z-10"
              style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}
            >
              {tooltip.text}
            </div>
          )}
          {/* Zoom controls */}
          <div className="absolute bottom-4 right-4 flex flex-col gap-1">
            {[
              { label: "+", delta: -1 },
              { label: "−", delta: 1 },
              { label: "⊙", delta: 0 },
            ].map(({ label, delta }) => (
              <button
                key={label}
                onClick={() => {
                  if (delta === 0) {
                    scaleRef.current = 1;
                    offsetRef.current = { x: (canvasRef.current?.width ?? 800) / 2, y: (canvasRef.current?.height ?? 600) / 2 };
                  } else {
                    const factor = delta < 0 ? 1.25 : 0.8;
                    const cx = (canvasRef.current?.width ?? 800) / 2;
                    const cy = (canvasRef.current?.height ?? 600) / 2;
                    const newScale = Math.max(0.08, Math.min(6, scaleRef.current * factor));
                    const ratio = newScale / scaleRef.current;
                    offsetRef.current = {
                      x: cx - (cx - offsetRef.current.x) * ratio,
                      y: cy - (cy - offsetRef.current.y) * ratio,
                    };
                    scaleRef.current = newScale;
                  }
                }}
                className="w-8 h-8 rounded-md border border-border bg-card text-foreground text-sm font-mono flex items-center justify-center hover:bg-muted transition-colors shadow-sm"
                title={delta < 0 ? "Zoom in" : delta > 0 ? "Zoom out" : "Reset zoom"}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Node detail panel — Neo4j style */}
        {selectedNode && (
          <div
            className="w-72 shrink-0 border-l border-border bg-card flex flex-col overflow-hidden"
            style={{ animation: "fadeReveal 0.25s var(--ease-out-expo) both" }}
          >
            {/* Header */}
            <div className="px-4 py-3 border-b border-border flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="h-3 w-3 rounded-full shrink-0"
                    style={{ background: nodeColor }}
                  />
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground capitalize">
                    {selectedNode.group.replace("_", " ")}
                  </span>
                </div>
                <p className="text-sm font-semibold text-foreground break-words leading-snug">
                  {selectedNode.name}
                </p>
              </div>
              <button
                onClick={() => { selectedRef.current = null; setSelectedNode(null); }}
                className="shrink-0 text-muted-foreground hover:text-foreground transition-colors text-lg leading-none mt-0.5"
                aria-label="Close panel"
              >
                ×
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              {/* Core properties */}
              <div className="px-4 py-3 border-b border-border">
                <p className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground mb-2">Properties</p>
                <dl className="space-y-2">
                  {[
                    ["id",   selectedNode.id],
                    ["type", selectedNode.type],
                  ].map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[80px_1fr] gap-1 text-xs">
                      <dt className="font-mono text-muted-foreground truncate">{k}</dt>
                      <dd className="text-foreground break-words">{String(v)}</dd>
                    </div>
                  ))}
                  {extraProps.map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[80px_1fr] gap-1 text-xs">
                      <dt className="font-mono text-muted-foreground truncate">{k}</dt>
                      <dd className="text-foreground break-words">
                        {typeof v === "object" ? JSON.stringify(v) : String(v ?? "—")}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>

              {/* Relationships */}
              {neighbors.length > 0 && (
                <div className="px-4 py-3">
                  <p className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground mb-2">
                    Relationships · {neighbors.length}
                  </p>
                  <ul className="space-y-1.5">
                    {neighbors.map((nb) => {
                      const link = data.links.find(
                        (l) =>
                          (l.source === selectedNode.id && l.target === nb.id) ||
                          (l.target === selectedNode.id && l.source === nb.id),
                      );
                      const nbColor = GROUP_COLOR[nb.group] || "#888";
                      return (
                        <li
                          key={nb.id}
                          className="flex items-start gap-2 text-xs cursor-pointer group rounded-md px-2 py-1.5 hover:bg-muted transition-colors"
                          onClick={() => { selectedRef.current = nb.id; setSelectedNode(nb); }}
                        >
                          <span className="h-2 w-2 rounded-full mt-0.5 shrink-0" style={{ background: nbColor }} />
                          <div className="min-w-0">
                            {link?.type && (
                              <span className="font-mono text-[9px] text-muted-foreground block">{link.type}</span>
                            )}
                            <span className="text-foreground group-hover:text-[var(--accent)] transition-colors break-words">
                              {nb.name}
                            </span>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
