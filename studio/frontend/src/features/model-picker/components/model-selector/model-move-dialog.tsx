// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { authFetch } from "@/features/auth";
import {
  bumpInventoryVersion,
  invalidateGgufVariantsCache,
} from "@/features/hub";
import { pickHuggingFaceCacheDir } from "@/features/native-intents";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { FolderBrowser } from "./folder-browser";

type Move = {
  phase:
    | "queued"
    | "checking"
    | "copying"
    | "verifying"
    | "finishing"
    | "completed"
    | "cancelled"
    | "failed";
  completed_bytes: number;
  total_bytes: number;
  destination: string;
  error: string | null;
  warning: string | null;
};
const active = (move: Move | null) =>
  move !== null && !["completed", "cancelled", "failed"].includes(move.phase);

const restorePhaseKeys = {
  queued: "modelStorage.restoreQueued",
  checking: "modelStorage.restoreChecking",
  copying: "modelStorage.restoreCopying",
  verifying: "modelStorage.restoreVerifying",
  finishing: "modelStorage.restoreFinishing",
  completed: "modelStorage.restoreCompleted",
  cancelled: "modelStorage.restoreCancelled",
  failed: "modelStorage.restoreFailed",
} as const;

async function requestMove(
  url: string,
  init?: RequestInit,
): Promise<Move | null> {
  const response = await authFetch(url, init);
  if (!response.ok)
    throw new Error(
      await readFastApiError(response, "Could not move the model"),
    );
  return response.json();
}

export function ModelMoveDialog({
  repoId,
  open,
  onOpenChange,
  mode = "move",
}: {
  repoId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode?: "move" | "restore";
}) {
  const t = useT();
  const restoring = mode === "restore";
  const [folder, setFolder] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [move, setMove] = useState<Move | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [loading, setLoading] = useState(true);
  const refreshed = useRef(false);
  const operationUrl = restoring
    ? "/api/hub/move-cached/restore"
    : "/api/hub/move-cached";
  const url = `${operationUrl}?repo_id=${encodeURIComponent(repoId)}`;
  const busy = active(move) || pending;
  useEffect(() => {
    if (!open) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    setLoading(true);
    const poll = async () => {
      try {
        const next = await requestMove(url);
        if (disposed) return;
        setMove(next);
        setLoading(false);
        if (next?.phase === "completed" && !refreshed.current) {
          refreshed.current = true;
          bumpInventoryVersion();
          invalidateGgufVariantsCache();
        }
        if (active(next)) refreshed.current = false;
      } catch (err) {
        if (!disposed) {
          setError(String(err instanceof Error ? err.message : err));
          setLoading(false);
        }
      }
      if (!disposed) timer = setTimeout(() => void poll(), 1000);
    };
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [open, url]);
  const choose = async () => {
    if (!isTauri) {
      setBrowserOpen(true);
      return;
    }
    try {
      const selected = await pickHuggingFaceCacheDir();
      if (selected) {
        setFolder(selected);
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const start = async () => {
    setPending(true);
    setError(null);
    refreshed.current = false;
    try {
      setMove(
        await requestMove(operationUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            restoring ? { repo_id: repoId } : { repo_id: repoId, folder },
          ),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(false);
    }
  };
  const cancel = async () => {
    setPending(true);
    try {
      setMove(
        await requestMove(
          `${operationUrl}/cancel?repo_id=${encodeURIComponent(repoId)}`,
          { method: "POST" },
        ),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(false);
    }
  };
  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent
          className="sm:max-w-lg"
          onClick={(event) => event.stopPropagation()}
        >
          <DialogHeader>
            <DialogTitle>
              {t(
                restoring
                  ? "modelStorage.restoreTitle"
                  : "modelStorage.title",
              )}
            </DialogTitle>
            <DialogDescription>
              {t(
                restoring
                  ? "modelStorage.restoreDescription"
                  : "modelStorage.description",
                { model: repoId },
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 text-sm">
            {restoring ? (
              <p className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                {t("modelStorage.restoreWarning")}
              </p>
            ) : (
              <>
                <p>{t("modelStorage.memory")}</p>
                <p className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                  {t("modelStorage.warning")}
                </p>
                <p className="text-muted-foreground">
                  {t("modelStorage.keepConnected")}
                </p>
                <Button
                  variant="outline"
                  disabled={busy || loading}
                  onClick={() => void choose()}
                >
                  {t("modelStorage.choose")}
                </Button>
                {folder && (
                  <p className="break-all font-mono text-xs">
                    {folder}/Unsloth Models
                  </p>
                )}
              </>
            )}
            {move && (
              <div aria-live="polite" className="space-y-2">
                <p>
                  {t(
                    restoring
                      ? restorePhaseKeys[move.phase]
                      : `modelStorage.${move.phase}`,
                  )}
                </p>
                {active(move) && (
                  <Progress
                    value={
                      move.total_bytes > 0
                        ? Math.min(
                            100,
                            (100 * move.completed_bytes) / move.total_bytes,
                          )
                        : 0
                    }
                  />
                )}
                {active(move) && move.total_bytes > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {Math.floor(
                      (100 * move.completed_bytes) / move.total_bytes,
                    )}
                    %
                  </p>
                )}
                {move.phase === "completed" && (
                  <p className="break-all font-mono text-xs">
                    {move.destination}
                  </p>
                )}
                {move.warning && <p role="alert">{move.warning}</p>}
              </div>
            )}
            {(error || move?.error) && (
              <p role="alert" className="text-destructive">
                {error || move?.error}
              </p>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              {t("common.close")}
            </Button>
            {busy ? (
              <Button
                variant="outline"
                disabled={pending || move?.phase === "finishing"}
                onClick={() => void cancel()}
              >
                {t("common.cancel")}
              </Button>
            ) : (
              <Button
                disabled={loading || (!restoring && !folder)}
                onClick={() => void start()}
              >
                {t(
                  restoring
                    ? "modelStorage.restoreAction"
                    : "modelStorage.action",
                )}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {!restoring && (
        <FolderBrowser
          open={browserOpen}
          onOpenChange={setBrowserOpen}
          onSelect={(path) => {
            setFolder(path);
            setError(null);
            setBrowserOpen(false);
          }}
          title={t("modelStorage.choose")}
          showModelHints={false}
        />
      )}
    </>
  );
}
