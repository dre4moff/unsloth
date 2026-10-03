// SPDX-License-Identifier: AGPL-3.0-only

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useLocale } from "@/i18n";
import { SmartphoneIcon, ZapIcon } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { selectAccelerationMode, prepareAcceleration } from "@/features/settings";
import { readyCompanionDevices, refreshCompanionChatStatus, useCompanionChatStore } from "../stores/companion-chat-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { getInferenceStatus } from "../api/chat-api";
import { applyActiveModelStatusToStore } from "../lib/apply-inference-status-to-store";
import { notifyModelLifecycle } from "@/lib/model-lifecycle-events";

export function IPhoneCompanionComposerButton({ side = "top" }: { side?: "top" | "bottom" }) {
  const italian = useLocale() === "it";
  const modeGroup = useId();
  const enabled = useCompanionChatStore((state) => state.enabled);
  const setEnabled = useCompanionChatStore((state) => state.setEnabled);
  const status = useCompanionChatStore((state) => state.status);
  const acceleration = useCompanionChatStore((state) => state.acceleration);
  const statusError = useCompanionChatStore((state) => state.statusError);
  const modelLoading = useChatRuntimeStore((state) => state.modelLoading);
  const supportsTools = useChatRuntimeStore((state) => state.supportsTools);
  const modelLoaded = useChatRuntimeStore((state) => Boolean(state.params.checkpoint));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [setup, setSetup] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [draftPath, setDraftPath] = useState("");
  const speed = acceleration?.mode === "speed";
  const ready = readyCompanionDevices(status).length > 0;
  const usable = !modelLoaded || supportsTools;
  const operational = speed ? acceleration?.ready : enabled && ready && usable;
  const cableAvailable = acceleration?.available === true;

  useEffect(() => {
    void refreshCompanionChatStatus();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void refreshCompanionChatStatus();
    }, 3_000);
    const refreshOnFocus = () => void refreshCompanionChatStatus();
    window.addEventListener("focus", refreshOnFocus);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", refreshOnFocus); };
  }, []);

  async function changeMode(mode: "agent" | "speed") {
    const runtime = useChatRuntimeStore.getState();
    const lease = runtime.beginModelLoading();
    if (!lease) return;
    notifyModelLifecycle({ runtime: "chat", model: runtime.params.checkpoint, loading: true });
    setBusy(true); setError(null);
    try {
      const next = await selectAccelerationMode(mode);
      useCompanionChatStore.getState().setAcceleration(next);
      await refreshCompanionChatStatus();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
      await refreshCompanionChatStatus();
    } finally {
      try {
        const current = await getInferenceStatus();
        runtime.endModelLoading(lease);
        if (current.active_model) applyActiveModelStatusToStore(current);
        else runtime.clearCheckpoint();
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
      runtime.endModelLoading(lease);
      notifyModelLifecycle({ runtime: "chat", model: runtime.params.checkpoint, loading: false });
      setBusy(false);
    }
  }

  async function prepare() {
    setBusy(true); setError(null);
    try {
      const next = await prepareAcceleration(modelPath || acceleration?.modelPath || "", draftPath || acceleration?.draftPath || "");
      useCompanionChatStore.getState().setAcceleration(next);
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setBusy(false); }
  }

  return (
    <Popover>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <button type="button" className="composer-pill-btn" data-pill-label="iPhone"
              data-active={operational ? "true" : "false"}
              aria-label={italian ? "Modalità iPhone Companion" : "iPhone Companion mode"}>
              <span className="composer-pill-glyph">
                {speed ? <ZapIcon className="size-[15px]" /> : <SmartphoneIcon className="size-[15px]" />}
                {enabled && !ready && !speed ? <span className="absolute right-0 top-0 size-1.5 rounded-full bg-amber-500 ring-1 ring-background" /> : null}
              </span>
              <span>{speed ? (italian ? "Velocità" : "Speed") : "iPhone"}</span>
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side={side} sideOffset={6}>
          {speed ? (italian ? "Backburner: lettura parallela e contesto su iPhone via USB." : "Backburner: parallel prefill and phone-held context over USB.")
            : (italian ? "Subagente iPhone o accelerazione via cavo." : "iPhone subagent or wired acceleration.")}
        </TooltipContent>
      </Tooltip>
      <PopoverContent side={side} align="start" className="w-80 space-y-3">
        <p className="text-sm font-medium">iPhone Companion</p>
        <div role="radiogroup" aria-label={italian ? "Modalità iPhone" : "iPhone mode"} className="space-y-2">
          <label className="flex items-center gap-2 text-sm">
            <input type="radio" name={modeGroup} checked={!speed} disabled={busy || modelLoading || acceleration?.preparing}
              onChange={() => { if (speed) void changeMode("agent"); }} />
            {italian ? "Agente" : "Agent"}
          </label>
          {cableAvailable || speed ? (
            <label className="flex items-center gap-2 text-sm">
              <input type="radio" name={modeGroup} checked={speed}
                disabled={busy || modelLoading || !cableAvailable || !acceleration?.prepared || acceleration?.preparing}
                onChange={() => void changeMode("speed")} />
              {italian ? "Aumenta velocità" : "Increase speed"}
              {cableAvailable ? <span className="ml-auto text-xs text-muted-foreground">USB {acceleration?.phone?.speedGbps} Gb/s</span> : null}
            </label>
          ) : null}
        </div>
        {!speed ? <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={enabled} disabled={!usable || busy || acceleration?.preparing} onChange={(event) => setEnabled(event.target.checked)} />
          {italian ? "Usa il subagente nelle chat" : "Use subagent in chats"}
        </label> : <p className="text-xs text-muted-foreground">{italian ? "Subagenti sospesi. Il motore originale usa una richiesta alla volta." : "Subagents paused. The original engine serves one request at a time."}</p>}
        {cableAvailable ? <>
          <p className="text-xs text-muted-foreground">{italian ? "Apri la pagina Velocità su iPhone. Profilo originale: Qwen3.8-27B IQ4_XS e DFlash2. Accelera soprattutto la lettura dei prompt." : "Open Speed on iPhone. Original profile: Qwen3.8-27B IQ4_XS and DFlash2. Mostly speeds up prompt reading."}</p>
          {!speed ? <Button variant="outline" size="sm" onClick={() => setSetup(!setup)}>{italian ? "Prepara iPhone" : "Prepare iPhone"}</Button> : null}
          {setup && !speed ? <div className="space-y-2">
            <label className="block text-xs">Qwen3.8-27B IQ4_XS GGUF
              <Input value={modelPath || acceleration?.modelPath || ""} onChange={(event) => setModelPath(event.target.value)} aria-label="Qwen3.8 GGUF path" />
            </label>
            <label className="block text-xs">dflash2-v2-q4km-self16.gguf
              <Input value={draftPath || acceleration?.draftPath || ""} onChange={(event) => setDraftPath(event.target.value)} placeholder="/…/dflash2-v2-q4km-self16.gguf" aria-label="DFlash2 GGUF path" />
            </label>
            <Button size="sm" disabled={busy || acceleration?.preparing || !acceleration?.ready || !(draftPath || acceleration?.draftPath)} onClick={() => void prepare()}>
              {italian ? "Prepara e copia via USB" : "Prepare and copy over USB"}
            </Button>
          </div> : null}
          {acceleration?.progress ? <p className="text-xs" role="status">{acceleration.progress}</p> : null}
          <a className="text-xs underline" href="https://github.com/StayLameBro/backburner" target="_blank" rel="noreferrer">Backburner · StayLameBro</a>
        </> : null}
        {busy ? <p className="text-xs" role="status">{italian ? "Cambio del motore in corso…" : "Switching engine…"}</p> : null}
        {error || acceleration?.error || statusError ? <p className="text-xs text-destructive" role="alert">{error || acceleration?.error || statusError}</p> : null}
      </PopoverContent>
    </Popover>
  );
}
