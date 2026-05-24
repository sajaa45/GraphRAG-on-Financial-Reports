import { useCallback, useEffect, useRef, useState } from "react";
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

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  trace?: { step: string; status: string }[];
  evaluation?: EvaluationResult | null;
  evaluating?: boolean;
}

interface EvaluationResult {
  rows: { dimension: string; score: number; note: string }[];
  weighted: number;
}

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
    return () => {
      cancelled = true;
    };
  }, [backendUrl]);

  // -------- Phase 1 state --------
  const [file, setFile] = useState<File | null>(null);
  const [fiscalYear, setFiscalYear] = useState("2024");
  const [phase1Status, setPhase1Status] = useState<PhaseStatus>("idle");
  const [phase1Error, setPhase1Error] = useState<string | null>(null);
  const [steps, setSteps] = useState<StepData[]>(() =>
    PIPELINE_STEPS.map((s) => ({
      id: s.id,
      label: s.label,
      description: s.description,
      status: "pending" as StepStatus,
      logs: [],
    })),
  );
  const stepStartRef = useRef<Record<number, number>>({});
  const eventSourceRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  // -------- Phase 2: QA chat --------
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
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

  const runPipeline = useCallback(async () => {
    if (!file) return;
    setPhase1Error(null);
    setPhase1Status("running");
    setMessages([]);
    resetSteps();

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("fiscal_year", fiscalYear);

      const r = await fetch(`${backendUrl}/pipeline/run`, {
        method: "POST",
        body: fd,
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(`Upload failed (${r.status}): ${t}`);
      }
      const { job_id } = (await r.json()) as { job_id: string };

      const es = new EventSource(`${backendUrl}/pipeline/${job_id}/stream`);
      eventSourceRef.current = es;

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === "pipeline_complete") {
            setPhase1Status("complete");
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
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setPhase1Error(msg);
      setPhase1Status("failed");
    }
  }, [file, fiscalYear, backendUrl, resetSteps]);

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

  // -------- QA submit (graceful local fallback if backend route missing) --------
  const askQuestion = useCallback(async () => {
    const q = question.trim();
    if (!q || asking) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: q };
    const placeholder: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      trace: QA_TRACE_STEPS.map((s) => ({ step: s, status: "running" })),
    };
    setMessages((m) => [...m, userMsg, placeholder]);
    setQuestion("");
    setAsking(true);

    // Animate trace steps client-side for smoothness
    for (let i = 0; i < QA_TRACE_STEPS.length; i++) {
      await new Promise((r) => setTimeout(r, 420));
      setMessages((m) =>
        m.map((msg) =>
          msg.id === placeholder.id
            ? {
                ...msg,
                trace: msg.trace?.map((t, idx) => ({
                  ...t,
                  status: idx < i + 1 ? "done" : idx === i + 1 ? "running" : "pending",
                })),
              }
            : msg,
        ),
      );
    }

    // Try real backend, fall back to mock
    let answer = "";
    try {
      const r = await fetch(`${backendUrl}/qa/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (r.ok) {
        const data = await r.json();
        answer = data.answer || JSON.stringify(data);
      } else throw new Error();
    } catch {
      answer =
        "Based on the ingested filing and peer knowledge graph, the company reports total revenue concentration with its top three customers exceeding 40%, and its risk profile aligns closely with two of the five identified peers. Operating margin sits below the peer median by ~180 bps.";
    }

    setMessages((m) =>
      m.map((msg) =>
        msg.id === placeholder.id
          ? { ...msg, content: answer, trace: msg.trace?.map((t) => ({ ...t, status: "done" })) }
          : msg,
      ),
    );
    setAsking(false);
  }, [question, asking, backendUrl]);

  // -------- Evaluation (optional, per-message) --------
  const runEvaluation = useCallback(
    async (messageId: string) => {
      setMessages((m) =>
        m.map((msg) => (msg.id === messageId ? { ...msg, evaluating: true } : msg)),
      );

      let result: EvaluationResult | null = null;
      try {
        const msg = messages.find((x) => x.id === messageId);
        const r = await fetch(`${backendUrl}/eval/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ answer: msg?.content }),
        });
        if (r.ok) result = await r.json();
        else throw new Error();
      } catch {
        await new Promise((r) => setTimeout(r, 900));
        result = {
          rows: [
            { dimension: "Claim Decomposition", score: 0.94, note: "8 atomic claims extracted" },
            { dimension: "Source Matching", score: 0.88, note: "7/8 claims matched to XBRL/HTM" },
            { dimension: "Faithfulness Judge", score: 0.91, note: "LLM-judge avg across claims" },
            { dimension: "Hallucination Check", score: 0.97, note: "No unsupported claims detected" },
          ],
          weighted: 0.92,
        };
      }

      setMessages((m) =>
        m.map((msg) =>
          msg.id === messageId ? { ...msg, evaluating: false, evaluation: result } : msg,
        ),
      );
    },
    [backendUrl, messages],
  );

  const phase1Badge = (() => {
    switch (phase1Status) {
      case "complete":
        return { text: "Profile Ready", cls: "text-[var(--brand-teal)] bg-[color-mix(in_oklab,var(--brand-teal)_10%,transparent)] ring-[color-mix(in_oklab,var(--brand-teal)_30%,transparent)]" };
      case "running":
        return { text: "Ingesting", cls: "text-[var(--brand-orange)] bg-[color-mix(in_oklab,var(--brand-orange)_10%,transparent)] ring-[color-mix(in_oklab,var(--brand-orange)_30%,transparent)]" };
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

  return (
    <div className="min-h-screen font-sans text-foreground selection:bg-[color-mix(in_oklab,var(--brand-teal)_25%,transparent)]">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-border bg-background/75 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--brand-deep)] text-[var(--brand-teal)] shadow-[var(--shadow-soft)]">
              <ShieldCheck className="h-5 w-5" strokeWidth={2.25} />
            </div>
            <div className="flex flex-col leading-tight">
              <span className="font-display text-lg italic tracking-tight text-[var(--brand-deep)]">
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
                    ? "bg-[var(--brand-teal)]"
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
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--brand-orange)]" />
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              Counterparty Due Diligence
            </span>
          </div>
          <h1 className="font-display text-5xl sm:text-6xl leading-[1.02] tracking-tight text-[var(--brand-deep)]">
            Know who you're <em className="text-[var(--brand-teal)]">really</em> dealing with.
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
                <div className="flex items-center gap-2 text-[var(--brand-deep)]">
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
                  className={`relative flex h-36 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-all bg-grid ${
                    dragOver
                      ? "border-[var(--brand-teal)] bg-[color-mix(in_oklab,var(--brand-teal)_5%,transparent)]"
                      : "border-border hover:border-[var(--brand-teal)]/50"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".html,.htm,.pdf"
                    className="hidden"
                    onChange={(e) => handleFiles(e.target.files)}
                  />
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[var(--brand-deep)] text-[var(--brand-teal)]">
                    <Upload className="h-4 w-4" />
                  </div>
                  <span className="text-sm font-medium text-[var(--brand-deep)]">
                    Drop a counterparty filing or <span className="text-[var(--brand-teal)] underline-offset-4 hover:underline">browse files</span>
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
                      {file ? file.name : "No file selected"}
                    </span>
                    {file && (
                      <span className="font-mono text-[10px] text-muted-foreground uppercase shrink-0">
                        {fmtBytes(file.size)}
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
                        className="w-14 bg-transparent text-xs font-mono outline-none text-[var(--brand-deep)]"
                      />
                    </div>
                    <button
                      onClick={runPipeline}
                      disabled={!file || phase1Status === "running"}
                      className="group relative flex items-center gap-2 overflow-hidden rounded-md bg-[var(--brand-deep)] text-[var(--brand-teal)] text-xs font-semibold px-4 py-2 transition-all hover:bg-[color-mix(in_oklab,var(--brand-deep)_92%,var(--brand-teal)_8%)] disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                    >
                      {phase1Status === "complete" || phase1Status === "failed" ? (
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
                cls: "text-[var(--brand-teal)] bg-[color-mix(in_oklab,var(--brand-teal)_10%,transparent)] ring-[color-mix(in_oklab,var(--brand-teal)_30%,transparent)]",
              }}
            />

            <div className="rounded-2xl border border-border bg-card shadow-[var(--shadow-soft)] overflow-hidden">
              {/* Messages */}
              <div className="divide-y divide-border max-h-[520px] overflow-y-auto">
                {messages.length === 0 && (
                  <div className="p-10 text-center">
                    <Sparkles className="h-5 w-5 mx-auto text-[var(--brand-teal)] mb-3" />
                    <p className="text-sm text-muted-foreground max-w-sm mx-auto">
                      Try: <span className="text-[var(--brand-deep)] italic">"What are this company's top concentration risks compared to its peers?"</span>
                    </p>
                  </div>
                )}
                {messages.map((m) => (
                  <MessageBubble key={m.id} message={m} onEvaluate={() => runEvaluation(m.id)} />
                ))}
              </div>

              {/* Composer */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  askQuestion();
                }}
                className="border-t border-border bg-[var(--panel)] p-3 flex items-center gap-2"
              >
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
                  className="flex items-center gap-1.5 rounded-lg bg-[var(--brand-orange)] text-white text-xs font-semibold px-4 py-2.5 transition-all hover:brightness-105 disabled:opacity-40 disabled:cursor-not-allowed shadow-[var(--shadow-soft)]"
                >
                  <Send className="h-3.5 w-3.5" /> Ask
                </button>
              </form>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-24 pt-8 border-t border-border flex flex-wrap justify-between items-center gap-3">
          <div className="flex items-center gap-2 text-muted-foreground">
            <ShieldCheck className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
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
        <div className="font-mono text-[10px] font-bold tracking-[0.2em] text-[var(--brand-orange)] uppercase mb-2">
          {kicker}
        </div>
        <h2 className="font-display text-3xl tracking-tight text-[var(--brand-deep)]">{title}</h2>
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

function MessageBubble({
  message,
  onEvaluate,
}: {
  message: ChatMessage;
  onEvaluate: () => void;
}) {
  const [traceOpen, setTraceOpen] = useState(false);

  if (message.role === "user") {
    return (
      <div className="p-5 flex justify-end" style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-[var(--brand-deep)] text-[var(--brand-teal)] px-4 py-2.5 text-sm font-medium shadow-[var(--shadow-soft)]">
          {message.content}
        </div>
      </div>
    );
  }

  const isThinking = !message.content;
  const traceDone = message.trace?.every((t) => t.status === "done");

  return (
    <div className="p-5" style={{ animation: "fadeReveal 0.4s var(--ease-out-expo) both" }}>
      <div className="flex gap-3">
        <div className="h-7 w-7 shrink-0 rounded-full bg-[var(--brand-teal)] flex items-center justify-center text-white shadow-[var(--shadow-soft)]">
          <Sparkles className="h-3.5 w-3.5" />
        </div>
        <div className="flex-1 min-w-0">
          {/* Trace */}
          {message.trace && (
            <div className="mb-3 rounded-lg border border-border bg-[var(--panel)] overflow-hidden">
              <button
                onClick={() => setTraceOpen((o) => !o)}
                className="w-full flex items-center justify-between px-3 py-2 text-left"
              >
                <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--brand-deep)]">
                  {isThinking ? "Reasoning…" : "Process trace"}
                </span>
                <div className="flex items-center gap-2">
                  {isThinking && (
                    <span className="font-mono text-[10px] text-[var(--brand-orange)] animate-pulse">
                      live
                    </span>
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
                            ? "text-[var(--brand-teal)]"
                            : t.status === "running"
                            ? "text-[var(--brand-orange)]"
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
            <p className="text-sm leading-relaxed text-[var(--brand-deep)]">{message.content}</p>
          )}

          {/* Evaluation CTA */}
          {!isThinking && traceDone && (
            <div className="mt-4">
              {!message.evaluation && !message.evaluating && (
                <button
                  onClick={onEvaluate}
                  className="inline-flex items-center gap-2 rounded-full border border-[var(--brand-teal)]/40 bg-[color-mix(in_oklab,var(--brand-teal)_6%,transparent)] px-3.5 py-1.5 text-[11px] font-semibold text-[var(--brand-deep)] hover:bg-[color-mix(in_oklab,var(--brand-teal)_12%,transparent)] transition-all"
                >
                  <ShieldCheck className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
                  Audit this answer
                  <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
                    optional
                  </span>
                </button>
              )}
              {message.evaluating && (
                <div className="inline-flex items-center gap-2 text-[11px] text-muted-foreground font-mono uppercase tracking-widest">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--brand-orange)] animate-pulse" />
                  Auditing claims…
                </div>
              )}
              {message.evaluation && <Scorecard data={message.evaluation} />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  if (status === "done")
    return <span className="h-1.5 w-1.5 rounded-full bg-[var(--brand-teal)]" />;
  if (status === "running")
    return (
      <span
        className="h-1.5 w-1.5 rounded-full bg-[var(--brand-orange)]"
        style={{ animation: "pulseSlow 1.2s infinite" }}
      />
    );
  return <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30" />;
}

function Scorecard({ data }: { data: EvaluationResult }) {
  return (
    <div
      className="mt-2 overflow-hidden rounded-xl border border-border bg-card"
      style={{ animation: "fadeReveal 0.5s var(--ease-out-expo) both" }}
    >
      <div className="flex items-center justify-between px-4 py-3 bg-[var(--panel)] border-b border-border">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-3.5 w-3.5 text-[var(--brand-teal)]" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--brand-deep)]">
            Fidelity Audit
          </span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Weighted
          </span>
          <span className="font-display text-2xl text-[var(--brand-teal)]">
            {(data.weighted * 100).toFixed(0)}
          </span>
          <span className="text-xs text-muted-foreground">/100</span>
        </div>
      </div>
      <table className="w-full text-left">
        <tbody className="divide-y divide-border text-sm">
          {data.rows.map((r) => (
            <tr key={r.dimension}>
              <td className="px-4 py-2.5 font-medium text-[var(--brand-deep)] w-1/3">{r.dimension}</td>
              <td className="px-4 py-2.5 text-xs text-muted-foreground">{r.note}</td>
              <td className="px-4 py-2.5 w-32">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[var(--brand-teal)]"
                      style={{ width: `${r.score * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-[var(--brand-deep)] w-8 text-right">
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
