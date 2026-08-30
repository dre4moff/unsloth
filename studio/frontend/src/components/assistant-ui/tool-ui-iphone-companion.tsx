// SPDX-License-Identifier: AGPL-3.0-only

"use client";

import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { SmartphoneIcon } from "lucide-react";
import { memo } from "react";

import { Spinner } from "@/components/ui/spinner";
import {
  ToolFallbackArgs,
  ToolFallbackContent,
  ToolFallbackResult,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

const IPhoneCompanionToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const input = (args ?? {}) as {
    action?: "submit" | "status" | "collect";
    job_id?: string;
    kind?: string;
    instruction?: string;
    media_policy?: string;
  };
  const isRunning = status?.type === "running";
  const action = input.action || "submit";
  const kind = (input.kind || "supporting task").replaceAll("_", " ");
  const label =
    action === "submit"
      ? `iPhone Companion · ${kind}`
      : action === "collect"
        ? "iPhone Companion · collect"
        : "iPhone Companion · status";

  return (
    <ToolFallbackRoot defaultOpen={isRunning}>
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
                ? "Dispatching private work to the paired iPhone…"
                : "Reading the iPhone job mailbox…"}
            </span>
          </div>
        ) : null}
        <ToolFallbackArgs
          argsText={JSON.stringify(
            {
              action,
              ...(input.job_id ? { job_id: input.job_id } : {}),
              kind: input.kind,
              ...(input.instruction ? { instruction: input.instruction } : {}),
              ...(input.media_policy
                ? { media_policy: input.media_policy }
                : {}),
            },
            null,
            2,
          )}
        />
        {!isRunning ? <ToolFallbackResult result={result} /> : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const IPhoneCompanionToolUI = memo(
  IPhoneCompanionToolUIImpl,
) as ToolCallMessagePartComponent;
