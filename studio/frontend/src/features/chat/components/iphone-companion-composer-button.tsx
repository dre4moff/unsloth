// SPDX-License-Identifier: AGPL-3.0-only

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useLocale } from "@/i18n";
import { SmartphoneIcon } from "lucide-react";
import { useEffect } from "react";

import {
  readyCompanionDevices,
  refreshCompanionChatStatus,
  useCompanionChatStore,
} from "../stores/companion-chat-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

export function IPhoneCompanionComposerButton({
  side = "top",
}: {
  side?: "top" | "bottom";
}) {
  const locale = useLocale();
  const italian = locale === "it";
  const enabled = useCompanionChatStore((state) => state.enabled);
  const setEnabled = useCompanionChatStore((state) => state.setEnabled);
  const status = useCompanionChatStore((state) => state.status);
  const statusError = useCompanionChatStore((state) => state.statusError);
  const modelLoaded = useChatRuntimeStore(
    (state) => Boolean(state.params.checkpoint) && !state.modelLoading,
  );
  const supportsTools = useChatRuntimeStore((state) => state.supportsTools);
  const usable = !modelLoaded || supportsTools;
  const devices = readyCompanionDevices(status);
  const ready = devices.length > 0;
  const operational = enabled && ready && usable;

  useEffect(() => {
    void refreshCompanionChatStatus();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        void refreshCompanionChatStatus();
      }
    }, 3_000);
    const refreshOnFocus = () => void refreshCompanionChatStatus();
    window.addEventListener("focus", refreshOnFocus);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refreshOnFocus);
    };
  }, []);

  const selection =
    status?.settings.mode === "multiple"
      ? italian
        ? `${devices.length} iPhone selezionati`
        : `${devices.length} selected iPhones`
      : devices[0]?.name ?? "iPhone";
  const detail = !usable
    ? italian
      ? "Il modello selezionato non supporta i tool di Studio."
      : "The selected model does not support Studio tools."
    : !enabled
      ? italian
        ? "Disattivato nelle chat."
        : "Disabled in chats."
      : ready
        ? italian
          ? `Subagent attivo · ${status?.settings.mode === "multiple" ? "Multi-iPhone" : "migliore automatico"}: ${selection}. Il modello Mac può delegargli task generali e resta orchestratore e fallback.`
          : `Subagent active · ${status?.settings.mode === "multiple" ? "Multi-iPhone" : "automatic best"}: ${selection}. The Mac model can delegate general tasks and remains orchestrator and fallback.`
        : statusError
          ? statusError
          : status?.settings.enabled === false
            ? italian
              ? "Companion è disattivato in Impostazioni → Connessioni."
              : "Companion is disabled in Settings → Connections."
            : italian
              ? "In attesa di un iPhone associato, connesso e pronto."
              : "Waiting for a paired, connected, ready iPhone.";

  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          className="composer-pill-btn"
          data-pill-label="iPhone"
          data-active={operational ? "true" : "false"}
          aria-label={
            enabled
              ? italian
                ? "Disattiva iPhone Companion nelle chat"
                : "Disable iPhone Companion in chats"
              : italian
                ? "Attiva iPhone Companion nelle chat"
                : "Enable iPhone Companion in chats"
          }
          aria-pressed={enabled}
          disabled={!usable}
          onClick={() => setEnabled(!enabled)}
        >
          <span className="composer-pill-glyph">
            <SmartphoneIcon className="size-[15px]" />
            {enabled && !ready ? (
              <span className="absolute right-0 top-0 size-1.5 rounded-full bg-amber-500 ring-1 ring-background" />
            ) : null}
          </span>
          <span>iPhone</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side={side} sideOffset={6} className="max-w-72">
        {detail}
      </TooltipContent>
    </Tooltip>
  );
}
