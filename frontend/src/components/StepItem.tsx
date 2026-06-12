import { Check, X } from "lucide-react";

export type StepStatus = "pending" | "running" | "done" | "failed";

export interface StepData {
  id: number | string;
  label: string;
  description?: string;
  status: StepStatus;
  summary?: string;
  durationMs?: number;
  logs?: string[];
}

export function StepItem({ step, isLast }: { step: StepData; isLast?: boolean }) {
  const { status, label, description, summary, durationMs, logs } = step;

  return (
    <div
      className={`relative flex gap-4 ${isLast ? "" : "pb-6"} ${
        status === "pending" ? "opacity-40" : ""
      }`}
    >
      <div className="relative z-10 mt-1 flex h-4 w-4 shrink-0 items-center justify-center">
        {status === "done" && (
          <div
            className="flex h-4 w-4 items-center justify-center rounded-full bg-success text-[10px] font-bold text-white ring-4 ring-background"
            style={{ animation: "checkPop 0.3s var(--ease-out-expo) both" }}
          >
            <Check className="h-2.5 w-2.5" strokeWidth={3} />
          </div>
        )}
        {status === "running" && (
          <div className="relative h-4 w-4 rounded-full border-2 border-accent bg-background ring-4 ring-background">
            <div
              className="absolute inset-0 rounded-full bg-accent"
              style={{ animation: "pulseSlow 1.5s infinite" }}
            />
          </div>
        )}
        {status === "pending" && (
          <div className="h-4 w-4 rounded-full border-2 border-border bg-background ring-4 ring-background" />
        )}
        {status === "failed" && (
          <div className="flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-white ring-4 ring-background">
            <X className="h-2.5 w-2.5" strokeWidth={3} />
          </div>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 flex-wrap">
          <h3 className="text-sm font-medium text-foreground">{label}</h3>
          {status === "running" && (
            <span className="font-mono text-[10px] text-accent animate-pulse">
              Running…
            </span>
          )}
          {status === "done" && durationMs != null && (
            <span className="font-mono text-[10px] text-muted-foreground">
              {(durationMs / 1000).toFixed(1)}s
            </span>
          )}
          {status === "failed" && (
            <span className="font-mono text-[10px] text-destructive">Failed</span>
          )}
        </div>

        {status === "done" && summary && (
          <p className="mt-1 text-xs text-muted-foreground">{summary}</p>
        )}
        {status === "pending" && description && (
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        )}
        {status === "running" && logs && logs.length > 0 && (
          <div className="mt-2 space-y-1.5 font-mono text-[10px] text-muted-foreground">
            {logs.slice(-3).map((log, i) => (
              <div key={i} className="flex gap-2">
                <span>&gt;</span>
                <span>{log}</span>
              </div>
            ))}
          </div>
        )}
        {status === "failed" && summary && (
          <p className="mt-1 text-xs text-destructive">{summary}</p>
        )}
      </div>
    </div>
  );
}