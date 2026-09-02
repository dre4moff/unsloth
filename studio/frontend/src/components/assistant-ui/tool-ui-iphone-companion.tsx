// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import {
  CheckCircle2Icon,
  CircleAlertIcon,
  Clock3Icon,
  SmartphoneIcon,
} from "lucide-react";
import { memo, useEffect, useMemo, useState } from "react";

import { Spinner } from "@/components/ui/spinner";
import {
  loadCompanionStatus,
  type ActiveCompanionTask,
} from "@/features/settings/api/companion";
import { cn } from "@/lib/utils";
import {
  ToolFallbackArgs,
  ToolFallbackContent,
  ToolFallbackResult,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

type CompanionInputItem = {
  id?: string;
  text?: string;
  instruction?: string;
};

type CompanionInput = {
  action?: "submit" | "status" | "collect" | "wait";
  job_id?: string;
  wait_seconds?: number;
  kind?: string;
  text?: string;
  instruction?: string;
  maximum_tokens?: number;
  media_policy?: string;
  items?: CompanionInputItem[];
};

type CompanionJobItem = {
  id?: string;
  taskID?: string;
  state?: string;
  index?: number;
  deviceID?: string;
  deviceName?: string;
  durationMS?: number;
  tokensGenerated?: number;
  result?: unknown;
  error?: string;
  text?: string;
  instruction?: string;
};

type CompanionJob = {
  jobID?: string;
  state?: string;
  kind?: string;
  taskIDs?: string[];
  items?: CompanionJobItem[];
  result?: CompanionJobItem;
  parallelResults?: CompanionJobItem[];
  error?: string;
};

type CompanionPayload = CompanionJob & {
  source?: string;
  jobs?: CompanionJob[];
  message?: string;
};

function parsePayload(result: unknown): CompanionPayload | null {
  if (result && typeof result === "object" && !Array.isArray(result)) {
    return result as CompanionPayload;
  }
  if (typeof result !== "string" || !result.trim().startsWith("{")) return null;
  try {
    const parsed: unknown = JSON.parse(result);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as CompanionPayload)
      : null;
  } catch {
    return null;
  }
}

function resultText(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") return value;
  if (typeof value === "object" && !Array.isArray(value)) {
    const text = (value as { text?: unknown }).text;
    if (typeof text === "string") return text;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function jobItems(job: CompanionJob): CompanionJobItem[] {
  if (job.items?.length) return job.items;
  if (job.parallelResults?.length) return job.parallelResults;
  if (job.result) return [job.result];
  return (job.taskIDs ?? []).map((taskID, index) => ({
    id: String(index + 1),
    taskID,
    state: job.state,
    index,
  }));
}

const terminalStates = new Set(["completed", "partial", "failed", "cancelled"]);

function StatePill({ state }: { state: string }) {
  const normalized = state.toLowerCase();
  const isDone = normalized === "completed";
  const isError = normalized === "failed" || normalized === "cancelled";
  const isRunning = normalized === "running";
  const label =
    normalized === "completed"
      ? "Completed"
      : normalized === "failed"
        ? "Failed"
        : normalized === "cancelled"
          ? "Cancelled"
          : normalized === "dispatched"
            ? "Dispatched"
            : normalized === "queued"
              ? "Queued"
              : "Running";
  const Icon = isDone
    ? CheckCircle2Icon
    : isError
      ? CircleAlertIcon
      : isRunning
        ? null
        : Clock3Icon;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
        isDone &&
          "border-emerald-500/25 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
        isError && "border-destructive/25 bg-destructive/10 text-destructive",
        !isDone && !isError &&
          "border-border bg-muted/45 text-muted-foreground",
      )}
    >
      {isRunning ? (
        <Spinner className="size-3" />
      ) : Icon ? (
        <Icon className="size-3" />
      ) : null}
      {label}
    </span>
  );
}

function shortID(value?: string): string {
  if (!value) return "pending";
  return value.length > 12 ? `${value.slice(0, 8)}…` : value;
}

const IPhoneCompanionToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const input = (args ?? {}) as CompanionInput;
  const payload = useMemo(() => parsePayload(result), [result]);
  const jobs = useMemo<CompanionJob[]>(() => {
    if (payload?.jobs) return payload.jobs;
    return payload?.jobID ? [payload] : [];
  }, [payload]);
  const taskIDs = useMemo(
    () =>
      Array.from(
        new Set(
          jobs.flatMap((job) => [
            ...(job.taskIDs ?? []),
            ...jobItems(job).flatMap((item) =>
              item.taskID ? [item.taskID] : [],
            ),
          ]),
        ),
      ),
    [jobs],
  );
  const pollKey = taskIDs.join("|");
  const shouldPoll = jobs.some(
    (job) => !terminalStates.has(job.state ?? "running"),
  );
  const [activeTasks, setActiveTasks] = useState<ActiveCompanionTask[]>([]);

  useEffect(() => {
    if (!pollKey || !shouldPoll) return;
    let disposed = false;
    let polls = 0;
    const wanted = new Set(pollKey.split("|"));
    const refresh = async () => {
      polls += 1;
      try {
        const companion = await loadCompanionStatus();
        if (!disposed) {
          setActiveTasks(
            companion.activeTasks.filter((task) => wanted.has(task.taskID)),
          );
        }
      } catch {
        // The tool result remains authoritative if settings polling is unavailable.
      }
    };
    void refresh();
    const interval = window.setInterval(() => {
      if (polls >= 150) {
        window.clearInterval(interval);
        return;
      }
      void refresh();
    }, 2000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, [pollKey, shouldPoll]);

  const activeByTaskID = useMemo(
    () => new Map(activeTasks.map((task) => [task.taskID, task])),
    [activeTasks],
  );
  const inputByID = useMemo(
    () =>
      new Map(
        (input.items ?? []).map((item, index) => [
          item.id ?? String(index),
          item,
        ]),
      ),
    [input.items],
  );
  const isRunning = status?.type === "running";
  const action = input.action || "submit";
  const itemCount = Math.max(
    input.items?.length ?? 0,
    jobs.reduce((count, job) => count + jobItems(job).length, 0),
  );
  const kind = (input.kind || jobs[0]?.kind || "subagent").replaceAll("_", " ");
  const label =
    action === "submit"
      ? `iPhone Companion · ${itemCount || 1} ${itemCount === 1 ? kind : "subagents"}`
      : action === "collect"
        ? "iPhone Companion · collected work"
        : action === "wait"
          ? "iPhone Companion · join"
          : "iPhone Companion · status";

  return (
    <ToolFallbackRoot defaultOpen={isRunning || itemCount > 1}>
      <ToolFallbackTrigger
        toolName={label}
        status={status}
        icon={SmartphoneIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-3.5" />
            <span>
              {action === "submit"
                ? "Submitting independent work to the iPhone pool…"
                : action === "wait"
                  ? "Joining completed iPhone subagent work…"
                  : "Reading the iPhone job mailbox…"}
            </span>
          </div>
        ) : null}

        {jobs.length ? (
          <div className="space-y-3">
            {jobs.map((job, jobIndex) => {
              const items = jobItems(job);
              return (
                <section
                  key={job.jobID ?? `job-${jobIndex}`}
                  className="overflow-hidden rounded-xl border border-border/75 bg-muted/15"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
                    <div className="min-w-0">
                      <div className="text-xs font-semibold text-foreground">
                        Job {shortID(job.jobID)}
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        {items.length} independent {items.length === 1 ? "task" : "tasks"}
                      </div>
                    </div>
                    <StatePill state={job.state ?? "running"} />
                  </div>
                  <div className="space-y-2 p-2">
                    {items.map((serverItem, itemIndex) => {
                      const prompt =
                        inputByID.get(serverItem.id ?? "") ??
                        input.items?.[itemIndex];
                      const active = serverItem.taskID
                        ? activeByTaskID.get(serverItem.taskID)
                        : undefined;
                      const output = resultText(serverItem.result);
                      const rawState = serverItem.error
                        ? "failed"
                        : active
                          ? "running"
                          : serverItem.state === "running"
                            ? "dispatched"
                            : (serverItem.state ?? job.state ?? "queued");
                      const deviceName = active?.deviceName ?? serverItem.deviceName;
                      return (
                        <article
                          key={
                            serverItem.taskID ??
                            `${serverItem.id}-${itemIndex}`
                          }
                          className="rounded-lg border border-border/70 bg-background/80 px-3 py-2.5 shadow-sm"
                        >
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="min-w-0">
                              <div className="truncate text-sm font-medium text-foreground">
                                {serverItem.id || `Subtask ${itemIndex + 1}`}
                              </div>
                              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[11px] text-muted-foreground">
                                <span>
                                  {deviceName
                                    ? `iPhone: ${deviceName}`
                                    : "Waiting for an iPhone"}
                                </span>
                                {serverItem.durationMS != null ? (
                                  <span>
                                    {(serverItem.durationMS / 1000).toFixed(1)}s
                                  </span>
                                ) : null}
                                {serverItem.tokensGenerated != null ? (
                                  <span>{serverItem.tokensGenerated} tokens</span>
                                ) : null}
                              </div>
                            </div>
                            <StatePill state={rawState} />
                          </div>
                          {prompt?.instruction ? (
                            <p className="mt-2 text-xs text-muted-foreground">
                              {prompt.instruction}
                            </p>
                          ) : null}
                          {serverItem.error ? (
                            <div className="mt-2 rounded-md bg-destructive/10 px-2.5 py-2 text-xs text-destructive">
                              {serverItem.error}
                            </div>
                          ) : null}
                          {output ? (
                            <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/60 bg-muted/25 p-2.5 font-sans text-xs leading-relaxed text-foreground">
                              {output}
                            </pre>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                </section>
              );
            })}
          </div>
        ) : null}

        <ToolFallbackArgs
          argsText={JSON.stringify(
            {
              action,
              ...(input.job_id ? { job_id: input.job_id } : {}),
              ...(input.wait_seconds
                ? { wait_seconds: input.wait_seconds }
                : {}),
              ...(input.kind ? { kind: input.kind } : {}),
              ...(input.maximum_tokens
                ? { maximum_tokens: input.maximum_tokens }
                : {}),
              ...(input.instruction
                ? { instruction: input.instruction }
                : {}),
              ...(input.items?.length
                ? {
                    items: input.items.map((item, index) => ({
                      id: item.id ?? String(index),
                      ...(item.instruction
                        ? { instruction: item.instruction }
                        : {}),
                    })),
                  }
                : {}),
              ...(input.media_policy
                ? { media_policy: input.media_policy }
                : {}),
            },
            null,
            2,
          )}
        />
        {!isRunning && !payload ? <ToolFallbackResult result={result} /> : null}
        {!isRunning && payload?.message ? (
          <p className="text-xs text-muted-foreground">{payload.message}</p>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const IPhoneCompanionToolUI = memo(
  IPhoneCompanionToolUIImpl,
) as ToolCallMessagePartComponent;
