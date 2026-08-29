// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TurnPlan } from "@/features/chat/types/api";

function StepMarker({
  status,
}: { status: TurnPlan["steps"][number]["status"] }) {
  if (status === "completed") {
    return (
      <span
        aria-hidden="true"
        className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-foreground text-background"
      >
        <span className="text-[12px] font-bold leading-none">✓</span>
      </span>
    );
  }
  return (
    <span
      aria-hidden="true"
      className={`mt-0.5 size-5 shrink-0 rounded-full border-[1.5px] ${
        status === "in_progress"
          ? "animate-pulse border-foreground shadow-[inset_0_0_0_4px_hsl(var(--background))] bg-foreground/25"
          : "border-foreground/55"
      }`}
    />
  );
}

export function TurnPlanCard({ plan }: { plan: TurnPlan }) {
  if (!Array.isArray(plan.steps) || plan.steps.length === 0) {
    return null;
  }
  const current = Math.min(
    Math.max(Number.isFinite(plan.current_step) ? plan.current_step : 0, 0),
    plan.steps.length - 1,
  );
  const shownStep =
    plan.status === "completed" ? plan.steps.length : current + 1;

  return (
    <section
      aria-label="Execution plan"
      aria-live="polite"
      className="mb-4 w-full max-w-2xl rounded-2xl border border-border/70 bg-muted/25 px-4 py-3.5"
    >
      <div className="mb-3 min-w-0">
        <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
          Objective
        </div>
        <div className="mt-1 line-clamp-2 text-sm font-medium text-foreground/90">
          {plan.objective}
        </div>
      </div>

      <ol className="space-y-2.5">
        {plan.steps.map((item, index) => (
          <li
            key={`${index}-${item.step}`}
            className={`flex min-w-0 items-start gap-3 text-[15px] leading-5 ${
              item.status === "completed"
                ? "text-muted-foreground"
                : item.status === "in_progress"
                  ? "font-medium text-foreground"
                  : "text-foreground/70"
            }`}
          >
            <StepMarker status={item.status} />
            <span
              className={
                item.status === "completed"
                  ? "line-through decoration-foreground/25"
                  : ""
              }
            >
              {item.step}
            </span>
          </li>
        ))}
      </ol>

      <div className="mt-3.5 flex justify-center border-t border-border/60 pt-3">
        <div className="inline-flex items-center gap-2 rounded-full border border-border bg-background/80 px-3 py-1.5 text-sm text-muted-foreground shadow-xs">
          <span
            aria-hidden="true"
            className={`size-3 rounded-full border-2 ${
              plan.status === "completed"
                ? "border-foreground bg-foreground"
                : "border-primary/35 border-t-primary animate-spin"
            }`}
          />
          <span>
            Step {shownStep} / {plan.steps.length}
          </span>
        </div>
      </div>
    </section>
  );
}
