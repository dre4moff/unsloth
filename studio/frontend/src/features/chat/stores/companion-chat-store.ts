// SPDX-License-Identifier: AGPL-3.0-only

import { create } from "zustand";
import { persist } from "zustand/middleware";

import {
  loadCompanionStatus,
  type CompanionDevice,
  type CompanionStatus,
} from "@/features/settings/api/companion";

const READY_STATES = new Set(["ready", "leased", "running"]);
let refreshInFlight: Promise<void> | null = null;

type CompanionChatState = {
  enabled: boolean;
  status: CompanionStatus | null;
  statusError: string | null;
  setEnabled: (enabled: boolean) => void;
  setStatus: (status: CompanionStatus | null, error?: string | null) => void;
};

export const useCompanionChatStore = create<CompanionChatState>()(
  persist(
    (set) => ({
      enabled: true,
      status: null,
      statusError: null,
      setEnabled: (enabled) => set({ enabled }),
      setStatus: (status, statusError = null) => set({ status, statusError }),
    }),
    {
      name: "unsloth_chat_iphone_companion",
      partialize: (state) => ({ enabled: state.enabled }),
      merge: (persisted, current) => ({
        ...current,
        enabled:
          (persisted as Partial<CompanionChatState> | undefined)?.enabled ??
          true,
      }),
    },
  ),
);

export function readyCompanionDevices(
  status: CompanionStatus | null,
): CompanionDevice[] {
  if (!status?.settings.enabled) return [];
  const selected = new Set(status.settings.selectedDeviceIDs);
  return status.devices.filter((device) => {
    if (
      !device.enabled ||
      !device.connected ||
      !device.authenticated ||
      !READY_STATES.has(device.state) ||
      device.lowPowerMode ||
      device.battery < 0.1 ||
      device.thermalState >= 2 ||
      !device.capabilities?.taskKinds.length
    ) {
      return false;
    }
    return status.settings.mode !== "multiple" || selected.has(device.deviceID);
  });
}

export async function refreshCompanionChatStatus(): Promise<void> {
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = (async () => {
    try {
      const status = await loadCompanionStatus();
      useCompanionChatStore.getState().setStatus(status);
    } catch (error) {
      const current = useCompanionChatStore.getState();
      current.setStatus(
        current.status,
        error instanceof Error ? error.message : String(error),
      );
    } finally {
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}
