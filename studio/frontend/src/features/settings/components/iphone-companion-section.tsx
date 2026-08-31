// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useLocale } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  cancelCompanionTask, confirmCompanionPairing, enableCompanionDevice, loadCompanionStatus,
  rejectCompanionPairing, renameCompanionDevice, revokeCompanionDevice,
  saveCompanionSettings, type CompanionDevice, type CompanionStatus,
} from "../api/companion";

const localized = (locale: string, en: string, italian: string) => locale === "it" ? italian : en;

export function IPhoneCompanionSection() {
  const locale = useLocale();
  const tx = (en: string, italian: string) => localized(locale, en, italian);
  const [status, setStatus] = useState<CompanionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try { setStatus(await loadCompanionStatus()); setError(null); }
    catch (value) { setError(value instanceof Error ? value.message : String(value)); }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [refresh]);

  const mutate = async (operation: () => Promise<unknown>) => {
    setBusy(true);
    try { await operation(); await refresh(); }
    catch (value) { toast.error(value instanceof Error ? value.message : String(value)); }
    finally { setBusy(false); }
  };

  const update = (patch: Partial<CompanionStatus["settings"]>) => {
    if (!status) return;
    void mutate(() => saveCompanionSettings({ ...status.settings, ...patch }));
  };

  return (
    <section className="space-y-5 rounded-xl border p-4" aria-labelledby="iphone-companion-heading">
      <div className="flex items-start justify-between gap-4">
        <div><h2 id="iphone-companion-heading" className="font-semibold">iPhone Companion</h2>
          <p className="text-muted-foreground text-sm">{tx("Use paired iPhones as real asynchronous local subagents. The Mac sees the live pool, orchestrates parallel work, and remains the fallback.", "Usa gli iPhone associati come veri subagent locali asincroni. Il Mac vede il pool in tempo reale, orchestra il lavoro parallelo e resta il fallback.")}</p></div>
        <Switch checked={status?.settings.enabled ?? false} disabled={!status || busy} onCheckedChange={(enabled) => update({ enabled })} aria-label={tx("Enable iPhone Companion", "Attiva iPhone Companion")} />
      </div>
      {error ? <p className="text-destructive text-sm">{error}</p> : null}
      {status?.settings.enabled ? (
        <>
          <div className="space-y-2">
            <div className="font-medium text-sm">{tx("Device selection", "Selezione dispositivi")}</div>
            <label className="flex items-center gap-2 text-sm"><input type="radio" checked={status.settings.mode === "automatic_best"} onChange={() => update({ mode: "automatic_best" })} />{tx("Automatically use every available iPhone, best first", "Usa automaticamente tutti gli iPhone disponibili, migliori per primi")}</label>
            <label className="flex items-center gap-2 text-sm"><input type="radio" checked={status.settings.mode === "multiple"} onChange={() => update({ mode: "multiple" })} />{tx("Use multiple selected iPhones in parallel", "Usa più iPhone selezionati in parallelo")}</label>
            <p className="text-muted-foreground text-xs">{tx("Both modes distribute independent work in parallel. Automatic mode uses the whole eligible pool; Multi-iPhone restricts it to your selected devices. One generation and its KV cache are never sharded.", "Entrambe le modalità distribuiscono lavoro indipendente in parallelo. La modalità automatica usa l'intero pool idoneo; Multi-iPhone lo limita ai dispositivi selezionati. Una singola generazione e la relativa KV cache non vengono mai suddivise.")}</p>
          </div>
          {status.pendingPairings.map((pairing) => (
            <div key={pairing.pairingID} className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3">
              <div className="font-medium">{tx("Pair", "Associa")} {pairing.deviceName}</div>
              <div className="my-2 font-mono text-3xl tracking-widest" aria-label={`${tx("Pairing code", "Codice di associazione")} ${pairing.code}`}>{pairing.code}</div>
              <p className="text-muted-foreground text-xs">{tx("Confirm only if these six digits match the iPhone.", "Conferma solo se queste sei cifre corrispondono a quelle sull’iPhone.")}</p>
              <div className="mt-3 flex gap-2"><Button size="sm" disabled={busy} onClick={() => void mutate(() => confirmCompanionPairing(pairing.pairingID))}>{tx("Confirm", "Conferma")}</Button><Button size="sm" variant="destructive" disabled={busy} onClick={() => void mutate(() => rejectCompanionPairing(pairing.pairingID))}>{tx("Reject", "Rifiuta")}</Button></div>
            </div>
          ))}
          <div className="space-y-3">
            {status.devices.length === 0 ? <p className="text-muted-foreground text-sm">{tx("No paired iPhone. Open Companion on the same local network.", "Nessun iPhone associato. Apri Companion sulla stessa rete locale.")}</p> : status.devices.map((device) => (
              <DeviceRow key={device.deviceID} device={device} selected={status.settings.selectedDeviceIDs.includes(device.deviceID)} multi={status.settings.mode === "multiple"} busy={busy}
                onSelect={(selected) => update({ selectedDeviceIDs: selected ? [...status.settings.selectedDeviceIDs, device.deviceID] : status.settings.selectedDeviceIDs.filter((id) => id !== device.deviceID) })}
                onMutate={mutate} />
            ))}
          </div>
          {status.activeTasks.length > 0 ? <div className="space-y-2">
            <div className="font-medium text-sm">{tx("Current activity", "Attività corrente")}</div>
            {status.activeTasks.map((task) => <div key={task.taskID} className="flex items-center justify-between gap-3 rounded-lg border p-3 text-sm">
              <div><div className="font-medium">{task.kind.replaceAll("_", " ")}</div><div className="text-muted-foreground text-xs">{task.deviceName} · {new Date(task.startedAt).toLocaleTimeString()}</div></div>
              <Button size="sm" variant="destructive" disabled={busy} onClick={() => { if (window.confirm(tx("Cancel this iPhone task? It will not be reassigned automatically.", "Annullare questo task iPhone? Non verrà riassegnato automaticamente."))) void mutate(() => cancelCompanionTask(task.taskID)); }}>{tx("Cancel", "Annulla")}</Button>
            </div>)}
          </div> : null}
          <div className="text-muted-foreground text-xs">{tx("Dedicated WSS listener", "Listener WSS dedicato")}: {status.listenerPort ?? "—"} · {tx("Certificate pin", "Pin certificato")}: {status.certificateSHA256?.slice(0, 16) ?? "—"}</div>
        </>
      ) : <p className="text-muted-foreground text-sm">{tx("Remote tasks are cancelled and transferred to the Mac when this switch is off.", "Quando lo switch è disattivato, i task remoti vengono annullati e trasferiti al Mac.")}</p>}
    </section>
  );
}

function DeviceRow({ device, selected, multi, busy, onSelect, onMutate }: { device: CompanionDevice; selected: boolean; multi: boolean; busy: boolean; onSelect: (value: boolean) => void; onMutate: (operation: () => Promise<unknown>) => Promise<void> }) {
  const locale = useLocale();
  const tx = (en: string, italian: string) => localized(locale, en, italian);
  const [name, setName] = useState(device.name);
  return <div className="space-y-3 rounded-lg border p-3">
    <div className="flex items-center gap-3">{multi ? <Checkbox checked={selected} onCheckedChange={(value) => onSelect(value === true)} aria-label={tx("Select device", "Seleziona dispositivo")} /> : null}<div className={`h-2.5 w-2.5 rounded-full ${device.connected ? "bg-green-500" : "bg-muted-foreground/40"}`} /><Input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} onBlur={() => { const trimmed = name.trim(); if (!trimmed) { setName(device.name); } else if (trimmed !== device.name) { setName(trimmed); void onMutate(() => renameCompanionDevice(device.deviceID, trimmed)); } }} /><Switch checked={device.enabled} disabled={busy} onCheckedChange={(enabled) => void onMutate(() => enableCompanionDevice(device.deviceID, enabled))} /></div>
    <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4"><span>{tx("State", "Stato")}: {device.state}</span><span>{tx("Battery", "Batteria")}: {Math.round(device.battery * 100)}%</span><span>{tx("Latency", "Latenza")}: {Math.round(device.latencyMS)} ms</span><span>{tx("Queue", "Coda")}: {device.queueDepth}</span><span>{tx("Score", "Punteggio")}: {device.score}</span><span>{tx("Thermal", "Termica")}: {device.thermalState}</span><span>{tx("Free storage", "Spazio libero")}: {formatBytes(device.storageFreeBytes)}</span><span>{tx("Model", "Modello")}: {device.capabilities?.selectedModelID ?? "—"}</span><span>Vision: {device.capabilities?.supportsVision ? "✓" : "—"}</span><span>Audio: {device.capabilities?.supportsAudio ? "✓" : "—"}</span></div>
    <Button size="sm" variant="destructive" disabled={busy} onClick={() => { if (window.confirm(tx("Revoke this iPhone?", "Revocare questo iPhone?"))) void onMutate(() => revokeCompanionDevice(device.deviceID)); }}>{tx("Revoke", "Revoca")}</Button>
  </div>;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  return new Intl.NumberFormat(undefined, { style: "unit", unit: "gigabyte", maximumFractionDigits: 1 }).format(bytes / 1_000_000_000);
}
