// SPDX-License-Identifier: AGPL-3.0-only

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useLocale } from "@/i18n";
import { SmartphoneIcon, ZapIcon } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { selectAccelerationMode, prepareAcceleration, loadAccelerationDrafts, type AccelerationDraft } from "@/features/settings";
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
  const [open, setOpen] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [draftPath, setDraftPath] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<AccelerationDraft[]>([]);
  const [draftsLoading, setDraftsLoading] = useState(false);
  const [draftsError, setDraftsError] = useState<string | null>(null);
  const [manualDraft, setManualDraft] = useState(false);
  const selectedDraft = draftPath ?? acceleration?.draftPath ?? "";
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

  useEffect(() => {
    if (!open || !setup || speed) return;
    let cancelled = false;
    let inFlight = false;
    async function refreshDrafts() {
      if (inFlight || document.visibilityState !== "visible") return;
      inFlight = true;
      setDraftsLoading(true);
      try {
        const downloaded = await loadAccelerationDrafts();
        if (!cancelled) { setDrafts(downloaded); setDraftsError(null); }
      } catch (failure) {
        if (!cancelled) setDraftsError(failure instanceof Error ? failure.message : String(failure));
      } finally {
        inFlight = false;
        if (!cancelled) setDraftsLoading(false);
      }
    }
    void refreshDrafts();
    const timer = window.setInterval(() => void refreshDrafts(), 5_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [open, setup, speed]);

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
      const next = await prepareAcceleration(modelPath || acceleration?.modelPath || "", selectedDraft);
      useCompanionChatStore.getState().setAcceleration(next);
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setBusy(false); }
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
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
      <PopoverContent side={side} align="start" className="w-96 max-w-[calc(100vw-2rem)] max-h-[75vh] overflow-y-auto space-y-3">
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
          <p className="text-xs text-muted-foreground">{italian ? "Apri la pagina Velocità su iPhone. Qwen3.8-27B e varianti compatibili (anche abliterate), con quantizzazioni supportate e DFlash2. Accelera soprattutto la lettura dei prompt." : "Open Speed on iPhone. Qwen3.8-27B and compatible derivatives (including abliterated), with supported quantizations and DFlash2. Mostly speeds up prompt reading."}</p>
          {!speed ? <Button variant="outline" size="sm" onClick={() => setSetup(!setup)}>{italian ? "Prepara iPhone" : "Prepare iPhone"}</Button> : null}
          {setup && !speed ? <div className="space-y-2">
            <label className="block text-xs">Qwen3.8-27B GGUF
              <Input value={modelPath || acceleration?.modelPath || ""} onChange={(event) => setModelPath(event.target.value)} aria-label="Qwen3.8 GGUF path" />
            </label>
            <label className="block text-xs">{italian ? "DFlash2 scaricato" : "Downloaded DFlash2"}
              <select className="mt-1 w-full rounded-md border bg-background p-2 text-sm" aria-label={italian ? "DFlash2 scaricato" : "Downloaded DFlash2"}
                value={manualDraft ? "__manual__" : selectedDraft} disabled={busy || acceleration?.preparing}
                onChange={(event) => {
                  const manual = event.target.value === "__manual__";
                  setManualDraft(manual);
                  if (!manual) setDraftPath(event.target.value);
                }}>
                <option value="">{italian ? "Seleziona un draft scaricato…" : "Select a downloaded draft…"}</option>
                {selectedDraft && !drafts.some((draft) => draft.path === selectedDraft) ? <option value={selectedDraft}>{italian ? "Draft salvato" : "Saved draft"}</option> : null}
                {drafts.map((draft) => <option key={draft.path} value={draft.path}>{draft.repository ? `${draft.repository.split("/")[0]} · ` : ""}{draft.name} · {Math.round(draft.sizeBytes / 1024 ** 2)} MiB</option>)}
                <option value="__manual__">{italian ? "Percorso locale manuale…" : "Manual local path…"}</option>
              </select>
            </label>
            {manualDraft ? <Input value={selectedDraft} onChange={(event) => setDraftPath(event.target.value)} placeholder="/…/draft.gguf" aria-label="DFlash2 GGUF path" /> : null}
            {draftsError ? <p className="text-xs text-destructive" role="alert">{draftsError}</p>
              : draftsLoading && drafts.length === 0 ? <p className="text-xs" role="status">{italian ? "Ricerca dei DFlash2 compatibili…" : "Finding compatible DFlash2 drafts…"}</p>
              : drafts.length === 0 ? <p className="text-xs text-muted-foreground">{italian ? "Nessun DFlash2 compatibile scaricato. Attendi il download oppure scegli un file locale." : "No compatible DFlash2 downloaded. Wait for the download or select a local file."}</p> : null}
            <p className="text-xs text-muted-foreground">{italian ? "Su Mac da 16 GB: cache q4 da 8k sul Mac, contesto restante su iPhone e draft sulla CPU se necessario. Il limite scelto (anche 50k) viene mantenuto; prestazioni da verificare via cavo." : "16 GB Macs: 8k q4 cache locally, remaining context on iPhone, CPU draft when needed. Your selected limit (including 50k) is preserved; wired performance needs verification."}</p>
            <Button size="sm" disabled={busy || acceleration?.preparing || !acceleration?.ready || !selectedDraft || !(modelPath || acceleration?.modelPath)} onClick={() => void prepare()}>
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
