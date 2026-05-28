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
} from "lucide-react";
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
  source_page?: number | string | null;
  section_title?: string | null;
  summary: string;
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

function Index() {
  // -------- Backend config --------
  const [backendUrl, setBackendUrlState] = useState(DEFAULT_BACKEND_URL);
  const [showSettings, setShowSettings] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    setBackendUrlState(getBackendUrl());
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${backendUrl}/docs`, { method: "GET" });
        if (!cancelled) setConnected(r.ok);
      } catch {
        if (!cancelled) setConnected(false);
      }
    })();
    return () => { cancelled = true; };
  }, [backendUrl]);

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
  const [previousFilename, setPreviousFilename] = useState<string | null>(() => {
    try { return localStorage.getItem("verdant_filename"); } catch { return null; }
  });
  const [savedJobId, setSavedJobId] = useState<string | null>(() => {
    try { return localStorage.getItem("verdant_job_id"); } catch { return null; }
  });

  // Restore completion from a previous session stored in localStorage
  const [phase1Status, setPhase1Status] = useState<PhaseStatus>(() => {
    try {
      return localStorage.getItem("verdant_phase1_status") === "complete" ? "complete" : "idle";
    } catch { return "idle"; }
  });
  const [phase1Error, setPhase1Error] = useState<string | null>(null);
  const [steps, setSteps] = useState<StepData[]>(() => {
    try {
      const saved = localStorage.getItem("verdant_steps");
      if (saved) return JSON.parse(saved) as StepData[];
    } catch { /* ignore */ }
    return PIPELINE_STEPS.map((s) => ({
      id: s.id,
      label: s.label,
      description: s.description,
      status: "pending" as StepStatus,
      logs: [],
    }));
  });
  const stepStartRef = useRef<Record<number, number>>({});
  const eventSourceRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  // -------- Phase 2: QA chat --------
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [reasoningEnabled, setReasoningEnabled] = useState(false);
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
    // Find the first step that isn't done — that's where we resume from.
    const firstIncompleteIdx = steps.findIndex((s) => s.status !== "done");
    const startFromStep = firstIncompleteIdx + 1; // convert 0-indexed → 1-indexed step ID

    // Resume is only possible from step 3+ (steps 1-2 require the original file).
    if (!savedJobId || firstIncompleteIdx <= 0 || startFromStep < 3) {
      runPipeline();
      return;
    }

    // Keep done steps as-is; reset incomplete steps to pending.
    setSteps((prev) =>
      prev.map((s) => ({
        ...s,
        status:     s.id < startFromStep ? s.status     : ("pending" as StepStatus),
        logs:       s.id < startFromStep ? s.logs       : [],
        summary:    s.id < startFromStep ? s.summary    : undefined,
        durationMs: s.id < startFromStep ? s.durationMs : undefined,
      })),
    );
    stepStartRef.current = {};
    setPhase1Error(null);
    setPhase1Status("running");

    try {
      const r = await fetch(`${backendUrl}/pipeline/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prev_job_id:     savedJobId,
          start_from_step: startFromStep,
          fiscal_year:     fiscalYear,
        }),
      });
      if (!r.ok) throw new Error(`Resume failed (${r.status}): ${await r.text()}`);
      const { job_id } = (await r.json()) as { job_id: string };
      try { localStorage.setItem("verdant_job_id", job_id); } catch { /* ignore */ }
      setSavedJobId(job_id);
      startListening(job_id, file);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setPhase1Error(msg);
      setPhase1Status("failed");
    }
  }, [steps, savedJobId, fiscalYear, backendUrl, file, runPipeline, startListening]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  // Smooth scroll to QA when Phase 1 completes
  useEffect(() => {
    if (phase1Status === "complete") {
      setTimeout(() => phase2Ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 400);
    }
  }, [phase1Status]);

  // -------- QA submit --------
  const askQuestion = useCallback(async () => {
    const q = question.trim();
    if (!q || asking) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: q };
    const placeholder: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      trace: QA_TRACE_STEPS.map((s) => ({ step: s, status: "pending" })),
    };
    setMessages((m) => [...m, userMsg, placeholder]);
    setQuestion("");
    setAsking(true);

    // Animate trace steps while the real request is in flight
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
      if (stepIdx >= QA_TRACE_STEPS.length) { clearInterval(stepTimer); return; }
      const current = stepIdx;
      setMessages((m) =>
        m.map((msg) =>
          msg.id === placeholder.id
            ? {
                ...msg,
                trace: msg.trace?.map((t, idx) => ({
                  ...t,
                  status: idx < current ? "done" : idx === current ? "running" : "pending",
                })),
              }
            : msg,
        ),
      );
      stepIdx++;
    }, 700);

    let answer = "";
    let citations: Record<string, CitationInfo> = {};
    let reasoning_trace: string | null = null;
    try {
      const r = await fetch(`${backendUrl}/qa/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, reasoning: reasoningEnabled }),
      });
      if (r.ok) {
        const data = await r.json();
        answer = data.answer || "";
        citations = data.citations || {};
        reasoning_trace = data.reasoning_trace || null;
      } else {
        answer = `Error ${r.status}: ${await r.text()}`;
      }
    } catch (e: unknown) {
      answer = e instanceof Error ? e.message : "Request failed";
    }

    clearInterval(stepTimer);
    setMessages((m) =>
      m.map((msg) =>
        msg.id === placeholder.id
          ? {
              ...msg,
              content: answer,
              citations,
              reasoning_trace,
              trace: msg.trace?.map((t) => ({ ...t, status: "done" })),
            }
          : msg,
      ),
    );
    setAsking(false);
  }, [question, asking, backendUrl, reasoningEnabled]);

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

  // Resume is available when the pipeline failed partway through and we have
  // a saved job_id with at least one completed step at step 3 or later.
  const firstIncompleteIdx = steps.findIndex((s) => s.status !== "done");
  const resumeFromStep     = firstIncompleteIdx + 1; // 1-indexed
  const canResume =
    phase1Status === "failed" &&
    !!savedJobId &&
    steps.some((s) => s.status === "done") &&
    resumeFromStep >= 3;

  return (
    <div className="min-h-screen font-sans text-foreground selection:bg-[color-mix(in_oklab,var(--accent)_25%,transparent)]">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-border bg-background/75 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--elevated)] text-[var(--accent)] shadow-[var(--shadow-soft)]">
              <ShieldCheck className="h-5 w-5" strokeWidth={2.25} />
            </div>
            <div className="flex flex-col leading-tight">
              <span className="font-sans text-lg italic tracking-tight text-foreground">
                Verdant
              </span>
              <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
                KYC Intelligence
              </span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5">
              <div
                className={`h-1.5 w-1.5 rounded-full ${
                  connected === null
                    ? "bg-muted-foreground/40"
                    : connected
                    ? "bg-[var(--accent)]"
                    : "bg-destructive"
                }`}
              />
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                {connected === null ? "Checking" : connected ? "Engine online" : "Engine offline"}
              </span>
            </div>
            <button
              onClick={() => setShowSettings((s) => !s)}
              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Settings"
            >
              <Settings className="h-4 w-4" />
            </button>
          </div>
        </div>
        {showSettings && (
          <div className="border-t border-border bg-card">
            <div className="mx-auto max-w-5xl px-6 py-3 flex items-center gap-3">
              <label className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                Backend URL
              </label>
              <input
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrlState(e.target.value)}
                onBlur={(e) => setBackendUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-xs font-mono outline-none focus:ring-2 focus:ring-ring"
              />
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
            Upload a counterparty filing and Verdant maps its financial profile, peers, and risk surface into a queryable knowledge graph — with every answer auditable down to its source.
          </p>
        </header>

        {/* PHASE 1 */}
        <section style={{ animation: "slideUp 0.6s var(--ease-out-expo) both" }}>
          <PhaseHeader
            kicker="01 — Ingestion"
            title="Counterparty Filing"
            subtitle="Drop a 10-K, annual report, or registration document. Verdant parses it into structured entities and a graph."
            badge={phase1Badge}
          />

          <div className="grid gap-6">
            {/* Upload card */}
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
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    handleFiles(e.dataTransfer.files);
                  }}
                  className={`relative flex h-36 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-all ${
                    dragOver
                      ? "border-[var(--accent)] bg-[color-mix(in_oklab,var(--accent)_5%,transparent)]"
                      : "border-border hover:border-[var(--accent)]/50"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".html,.htm,.pdf"
                    className="hidden"
                    onChange={(e) => handleFiles(e.target.files)}
                  />
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
                      {file
                        ? file.name
                        : previousFilename && phase1Status === "complete"
                        ? previousFilename
                        : "No file selected"}
                    </span>
                    {file && (
                      <span className="font-mono text-[10px] text-muted-foreground uppercase shrink-0">
                        {fmtBytes(file.size)}
                      </span>
                    )}
                    {!file && previousFilename && phase1Status === "complete" && (
                      <span className="font-mono text-[10px] text-[var(--accent)] uppercase shrink-0">
                        restored
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-2 rounded-md border border-input bg-background px-2.5 py-1.5">
                      <label className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                        FY
                      </label>
                      <input
                        type="text"
                        value={fiscalYear}
                        onChange={(e) => setFiscalYear(e.target.value)}
                        className="w-14 bg-transparent text-xs font-mono outline-none text-foreground"
                      />
                    </div>
                    <button
                      onClick={canResume ? resumePipeline : runPipeline}
                      disabled={(canResume ? false : !file) || phase1Status === "running"}
                      className="group relative flex items-center gap-2 overflow-hidden rounded-md bg-[var(--elevated)] text-[var(--accent)] text-xs font-semibold px-4 py-2 transition-all hover:bg-[var(--elevated)] disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                    >
                      {canResume ? (
                        <>
                          <RotateCcw className="h-3.5 w-3.5" /> Resume from step {resumeFromStep}
                        </>
                      ) : phase1Status === "complete" || phase1Status === "failed" ? (
                        <>
                          <RotateCcw className="h-3.5 w-3.5" /> Re-ingest
                        </>
                      ) : (
                        <>
                          <Play className="h-3.5 w-3.5 fill-current" /> Run Diligence
                        </>
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

            {/* Stepper */}
            {(phase1Status !== "idle" || steps.some((s) => s.status !== "pending")) && (
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
                  <MessageBubble key={m.id} message={m} onEvaluate={(testType) => runEvaluation(m.id, testType)} />
                ))}
              </div>

              {/* Composer */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  askQuestion();
                }}
                className="border-t border-border bg-[var(--panel)] p-3 flex flex-col gap-2"
              >
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask about risks, financials, peer comparisons…"
                    disabled={asking}
                    className="flex-1 rounded-lg bg-card border border-input px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-ring transition-all placeholder:text-muted-foreground"
                  />
                  <button
                    type="submit"
                    disabled={!question.trim() || asking}
                    className="flex items-center gap-1.5 rounded-lg bg-[var(--warning)] text-white text-xs font-semibold px-4 py-2.5 transition-all hover:brightness-105 disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                  >
                    <Send className="h-3.5 w-3.5" /> Ask
                  </button>
                </div>
                <div className="flex items-center gap-2 px-1">
                  <button
                    type="button"
                    onClick={() => setReasoningEnabled((v) => !v)}
                    className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest transition-all ${
                      reasoningEnabled
                        ? "border-[var(--accent)]/50 bg-[color-mix(in_oklab,var(--accent)_8%,transparent)] text-[var(--accent)]"
                        : "border-border text-muted-foreground hover:border-muted-foreground"
                    }`}
                  >
                    <Sparkles className="h-3 w-3" />
                    {reasoningEnabled ? "Reasoning on" : "Reasoning off"}
                  </button>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    {reasoningEnabled ? "Chain-of-thought will appear below the answer" : "Enable to see step-by-step reasoning"}
                  </span>
                </div>
              </form>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-24 pt-8 border-t border-border flex flex-wrap justify-between items-center gap-3">
          <div className="flex items-center gap-2 text-muted-foreground">
            <ShieldCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
            <span className="text-[10px] font-mono uppercase tracking-widest">
              Verdant · KYC Engine · Auditable by design
            </span>
          </div>
          <div className="text-[10px] text-muted-foreground font-mono">
            backend · {backendUrl.replace(/^https?:\/\//, "")}
          </div>
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
}: {
  text: string;
  citations?: Record<string, CitationInfo>;
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
      return (
        <div key={id} className="flex gap-2 text-[10px] font-mono text-muted-foreground">
          <span className="shrink-0 text-[var(--accent)]">[{idx + 1}]</span>
          <span className="truncate">
            <span className="text-foreground">{info.company}</span>
            {" · "}
            <span className={info.role === "target" ? "text-[var(--accent)]" : "text-[var(--warning)]"}>
              {info.role}
            </span>
            {info.section_title && (
              <> · <span className="text-muted-foreground">{info.section_title}</span></>
            )}
            {info.source_page != null && (
              <> · <span className="text-foreground">p.{info.source_page}</span></>
            )}
            {" · "}
            {info.summary.slice(0, 80)}
            {href && (
              <>
                {" "}
                <a href={href} target="_blank" rel="noopener noreferrer"
                   className="text-[var(--accent)] underline underline-offset-2 hover:opacity-80">
                  source ↗
                </a>
              </>
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
// SourcesList — always-visible citations panel
// ---------------------------------------------------------------------------

function SourcesList({ citations }: { citations?: Record<string, CitationInfo> }) {
  if (!citations || Object.keys(citations).length === 0) return null;
  const entries = Object.entries(citations);
  return (
    <div className="mt-4 rounded-xl border border-border overflow-hidden">
      <div className="px-3 py-2 bg-[var(--panel)] border-b border-border flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
        <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
          Sources · {entries.length}
        </span>
      </div>
      <div className="divide-y divide-border">
        {entries.map(([id, info], idx) => {
          const href = info.document_url || null;
          return (
            <div key={id} className="flex gap-3 px-3 py-2.5 text-[11px]">
              <span className="shrink-0 font-mono text-[var(--accent)] w-5 pt-px">[{idx + 1}]</span>
              <div className="flex-1 min-w-0 space-y-0.5">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                  <span className="font-medium text-foreground">{info.company}</span>
                  <span className={`font-mono text-[9px] uppercase tracking-widest px-1.5 py-0.5 rounded-full ${
                    info.role === "target"
                      ? "bg-[color-mix(in_oklab,var(--accent)_10%,transparent)] text-[var(--accent)]"
                      : "bg-[color-mix(in_oklab,var(--warning)_10%,transparent)] text-[var(--warning)]"
                  }`}>
                    {info.role}
                  </span>
                  <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground/60">
                    {info.type}
                  </span>
                  {info.section_title && (
                    <span className="text-[10px] text-muted-foreground truncate max-w-[200px]">
                      {info.section_title}
                    </span>
                  )}
                  {info.source_page != null && (
                    <span className="font-mono text-[9px] text-foreground bg-muted px-1.5 py-0.5 rounded">
                      p.{info.source_page}
                    </span>
                  )}
                </div>
                <p className="text-muted-foreground leading-relaxed line-clamp-2">{info.summary}</p>
                {href && (
                  <a href={href} target="_blank" rel="noopener noreferrer"
                     className="inline-flex items-center gap-1 text-[10px] font-mono text-[var(--accent)] hover:underline underline-offset-2">
                    Open filing ↗
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ReasoningTrace — Claude-style shaded chain-of-thought
// ---------------------------------------------------------------------------

function ReasoningTrace({ trace }: { trace: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mt-4 rounded-xl overflow-hidden border border-[color-mix(in_oklab,var(--accent)_20%,transparent)] bg-[color-mix(in_oklab,var(--accent)_4%,transparent)]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-left"
      >
        <div className="flex items-center gap-2">
          <Sparkles className="h-3 w-3 text-[var(--accent)]/60" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--accent)]/70">
            Reasoning trace
          </span>
        </div>
        <ChevronDown className={`h-3.5 w-3.5 text-[var(--accent)]/50 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="px-4 pb-4 border-t border-[color-mix(in_oklab,var(--accent)_15%,transparent)]">
          <pre className="mt-3 whitespace-pre-wrap text-[11px] font-mono text-muted-foreground leading-relaxed max-h-80 overflow-y-auto">
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
}: {
  message: ChatMessage;
  onEvaluate: (testType: string) => void;
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
  const hasCitations = message.citations && Object.keys(message.citations).length > 0;

  return (
    <div className="p-5" style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
      <div className="flex gap-3">
        <div className="h-7 w-7 shrink-0 rounded-full bg-[var(--accent)] flex items-center justify-center text-white shadow-[var(--shadow-soft)]">
          <Sparkles className="h-3.5 w-3.5" />
        </div>
        <div className="flex-1 min-w-0">

          {/* Process trace — collapsible, auto-open while thinking */}
          {message.trace && (
            <div className="mb-4 rounded-lg border border-border bg-[var(--panel)] overflow-hidden">
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
                      <span className={
                        t.status === "done" ? "text-[var(--accent)]"
                        : t.status === "running" ? "text-[var(--warning)]"
                        : "text-muted-foreground/60"
                      }>
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
            <CitedText text={message.content} citations={message.citations} />
          )}

          {/* Reasoning trace — shaded, shown below answer */}
          {message.reasoning_trace && (
            <ReasoningTrace trace={message.reasoning_trace} />
          )}

          {/* Sources — always visible if any citations exist */}
          {!isThinking && hasCitations && (
            <SourcesList citations={message.citations} />
          )}

          {/* Evaluation */}
          {!isThinking && traceDone && (
            <div className="mt-4 rounded-xl border border-border overflow-hidden">
              <div className="px-4 py-3 bg-[var(--panel)] border-b border-border flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    Audit
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={selectedEval}
                    onChange={(e) => setSelectedEval(e.target.value)}
                    className="rounded-md border border-input bg-background px-2 py-1 text-[11px] font-mono outline-none focus:ring-1 focus:ring-ring text-foreground"
                  >
                    {EVAL_TESTS.map((t) => (
                      <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => onEvaluate(selectedEval)}
                    disabled={message.evaluating}
                    className="inline-flex items-center gap-1.5 rounded-md bg-[var(--elevated)] border border-[var(--accent)]/30 px-3 py-1.5 text-[11px] font-semibold text-[var(--accent)] hover:bg-[color-mix(in_oklab,var(--accent)_8%,transparent)] transition-all disabled:opacity-50"
                  >
                    {message.evaluating ? (
                      <>
                        <span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)] animate-pulse" />
                        Running…
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="h-3 w-3" /> Run
                      </>
                    )}
                  </button>
                </div>
              </div>
              {message.evaluation && <Scorecard data={message.evaluation} />}
              {!message.evaluation && !message.evaluating && (
                <div className="px-4 py-3 text-[11px] text-muted-foreground font-mono">
                  Select a test and click Run to audit this answer.
                </div>
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
  const pct = Math.round(data.weighted * 100);
  const scoreColor =
    pct >= 75 ? "text-[var(--accent)]" : pct >= 50 ? "text-[var(--warning)]" : "text-destructive";
  const barColor =
    pct >= 75 ? "bg-[var(--accent)]" : pct >= 50 ? "bg-[var(--warning)]" : "bg-destructive";

  return (
    <div style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
      {/* Overall score banner */}
      <div className="flex items-center gap-4 px-4 py-3 border-b border-border">
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
            <span className={`font-sans text-xl font-semibold ${scoreColor}`}>
              {pct}<span className="text-xs text-muted-foreground font-mono">/100</span>
            </span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full ${barColor} transition-all`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
      {/* Dimension rows */}
      {data.rows.length > 0 && (
        <table className="w-full text-left">
          <tbody className="divide-y divide-border">
            {data.rows.map((r, i) => (
              <tr key={i} className="hover:bg-muted/30 transition-colors">
                <td className="px-4 py-2.5 text-[11px] font-medium text-foreground w-1/3 truncate max-w-[160px]" title={r.dimension}>
                  {r.dimension}
                </td>
                <td className="px-4 py-2.5 text-[11px] text-muted-foreground leading-snug">{r.note}</td>
                <td className="px-4 py-2.5 w-28">
                  <div className="flex items-center gap-2">
                    <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          r.score >= 0.75 ? "bg-[var(--accent)]"
                          : r.score >= 0.5 ? "bg-[var(--warning)]"
                          : "bg-destructive"
                        }`}
                        style={{ width: `${r.score * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-[10px] text-foreground w-7 text-right shrink-0">
                      {Math.round(r.score * 100)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
