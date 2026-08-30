// SPDX-License-Identifier: AGPL-3.0-only
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type CompanionMode = "automatic_best" | "multiple";
export type CompanionSettings = { enabled: boolean; mode: CompanionMode; selectedDeviceIDs: string[] };
export type CompanionCapabilities = {
  protocolVersion: number; deviceID: string; deviceName: string; runtimeVersion: string;
  taskKinds: string[]; selectedModelID: string | null; supportsVision: boolean;
  supportsAudio: boolean; contextSize: number; maxRawMediaBytes: number;
};
export type CompanionDevice = {
  deviceID: string; name: string; enabled: boolean; connected: boolean; authenticated: boolean;
  battery: number; lowPowerMode: boolean; thermalState: number; state: string; latencyMS: number;
  queueDepth: number; storageFreeBytes: number; score: number; capabilities: CompanionCapabilities | null;
  lastSeen: string | null;
};
export type PendingPairing = {
  pairingID: string; deviceID: string; deviceName: string; signingPublicKey: string;
  code: string; createdAt: string; phoneConfirmed: boolean; desktopConfirmed: boolean;
};
export type ActiveCompanionTask = {
  taskID: string; deviceID: string; deviceName: string; kind: string; startedAt: string;
};
export type CompanionStatus = {
  settings: CompanionSettings; devices: CompanionDevice[]; pendingPairings: PendingPairing[];
  activeTasks: ActiveCompanionTask[];
  listenerPort: number | null; certificateSHA256: string | null;
};

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await authFetch(`/api/companion${path}`, init);
  if (!response.ok) throw new Error(await readFastApiError(response, "Companion request failed"));
  return response;
}

export async function loadCompanionStatus(): Promise<CompanionStatus> {
  return (await request("/status")).json();
}
export async function saveCompanionSettings(value: CompanionSettings): Promise<CompanionStatus> {
  return (await request("/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) })).json();
}
export async function confirmCompanionPairing(id: string): Promise<void> { await request(`/pairings/${id}/confirm`, { method: "POST" }); }
export async function rejectCompanionPairing(id: string): Promise<void> { await request(`/pairings/${id}/reject`, { method: "POST" }); }
export async function revokeCompanionDevice(id: string): Promise<void> { await request(`/devices/${id}`, { method: "DELETE" }); }
export async function renameCompanionDevice(id: string, name: string): Promise<void> {
  await request(`/devices/${id}/name`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
}
export async function enableCompanionDevice(id: string, enabled: boolean): Promise<void> {
  await request(`/devices/${id}/enabled`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled }) });
}
export async function cancelCompanionTask(id: string): Promise<void> { await request(`/tasks/${id}/cancel`, { method: "POST" }); }
