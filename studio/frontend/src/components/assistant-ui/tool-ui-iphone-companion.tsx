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
    kind?: string;
    instruction?: string;
    media_policy?: string;
  };
  const isRunning = status?.type === "running";
  const kind = (input.kind || "supporting task").replaceAll("_", " ");

  return (
    <ToolFallbackRoot defaultOpen={isRunning}>
      <ToolFallbackTrigger
        toolName={`iPhone Companion · ${kind}`}
        status={status}
        icon={SmartphoneIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-3.5" />
            <span>Running privately on the paired iPhone…</span>
          </div>
        ) : null}
        <ToolFallbackArgs
          argsText={JSON.stringify(
            {
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
