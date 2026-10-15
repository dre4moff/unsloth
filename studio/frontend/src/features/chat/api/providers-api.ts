// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import forge from "node-forge";
import { authFetch } from "@/features/auth/api";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type { ModelCatalogSnapshotEntry } from "../model-catalog-snapshot";
import type { CustomReasoningConfig } from "../custom-reasoning";


export type ProviderAuthKind = "api_key" | "chatgpt_oauth";
export type ProviderAuthStatus =
  | "disconnected"
  | "connected"
  | "reauthorization_required";

export interface ProviderCapabilities {
  supports_streaming: boolean;
  supports_tool_calling: boolean;
  supports_reasoning: boolean;
  supports_vision: boolean;
  supports_images: boolean;
  context_length?: number | null;
  api_mode: "chat_completions" | "responses" | "anthropic_messages";
}

export interface KaggleTPUManagedConfig {
  lab_path?: string | null;
  auto_start: boolean;
  auto_stop: boolean;
  keepalive_minutes: number;
  text_only: boolean;
  fast_start: boolean;
  max_num_seqs: number;
  mtp_tokens: number;
  reasoning_effort_default: "xhigh" | "medium" | "low";
}

export interface KaggleTPULifecycleStatus {
  provider_id: string;
  state:
    | "STOPPED"
    | "STARTING"
    | "PROVISIONING"
    | "LOADING"
    | "READY"
    | "DISCONNECTED"
    | "ERROR"
    | "STOPPING";
  message: string;
  base_url?: string | null;
  model?: string | null;
  context_length?: number | null;
  updated_at: string;
}

export interface ProviderRegistryEntry {
  provider_type: string;
  display_name: string;
  base_url: string;
  default_models: string[];

  model_capabilities?: Record<string, { vision?: boolean; studio_tools?: boolean }>;
  supports_streaming: boolean;
  supports_vision: boolean;
  supports_tool_calling: boolean;
  supports_reasoning?: boolean;
  supports_images?: boolean;
  context_length?: number | null;
  api_mode?: ProviderCapabilities["api_mode"];
  /** Studio runs its own tool loop (search/code/MCP/RAG) against this provider. */
  /** Unsloth runs its own tool loop (search/code/MCP/RAG) against this provider. */
  supports_studio_tools?: boolean;
  /** Backend-only entry, surfaced through a custom preset rather than the dropdown. */
  hidden?: boolean;
  /** remote = fetch /models; curated = huge catalogs — UI uses defaults + manual IDs only */
  model_list_mode?: "remote" | "curated";

  auth_kind?: ProviderAuthKind;
  base_url_editable?: boolean;
  model_ids_editable?: boolean;
}

export type ProviderApiType = "chat_completions" | "responses";
export type ConnectionApiType = ProviderApiType | "systemone";

export interface ProviderConfig {
  api_type?: ConnectionApiType;
  id: string;
  provider_type: string;
  display_name: string;
  base_url: string;
  is_enabled: boolean;

  has_api_key: boolean;
  has_kaggle_api_token?: boolean;

  auth_kind?: ProviderAuthKind;
  auth_status?: ProviderAuthStatus;
  models?: string[];
  available_models?: string[];
  max_output_tokens?: number | null;
  capabilities?: ProviderCapabilities;
  managed_config?: KaggleTPUManagedConfig | null;
  reasoning_config?: CustomReasoningConfig | null;
  created_at: string;
  updated_at: string;
}

export interface ProviderModelInfo {
  id: string;
  display_name: string;
  context_length?: number | null;
  owned_by?: string | null;
  /** Only the ChatGPT plan catalog reports this; the registry describes the rest. */
  vision?: boolean | null;
  /** Only Ollama's /api/tags reports these; absent is not the same as none. */
  capabilities?: string[] | null;
}

export interface ProviderModelReasoningInfo {
  supported_efforts?: string[] | null;
  mandatory?: boolean | null;
  default_effort?: string | null;
  default_enabled?: boolean | null;
}

export interface ProviderModelCapabilityInfo {
  id: string;
  input_modalities?: string[] | null;
  reasoning?: ProviderModelReasoningInfo | null;
  max_output_tokens?: number | null;
  supported_parameters?: string[] | null;
}

export interface ModelCatalogResponse {
  fetched_at: number;
  providers: Record<string, Record<string, ModelCatalogSnapshotEntry>>;
}

export function conservativeDiscoveredContextLength(
  models: readonly ProviderModelInfo[],
): number | null {
  const declared = models
    .map((model) => model.context_length)
    .filter(
      (value): value is number =>
        typeof value === "number" && Number.isSafeInteger(value) && value > 0,
    );
  return declared.length > 0 ? Math.min(...declared) : null;
}

export interface ProviderTestResult {
  success: boolean;
  message: string;
  models_count?: number | null;
}

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    const formatted = formatFastApiDetail(detail);
    if (formatted) return formatted;
    const message = (body as { message?: unknown }).message;
    if (typeof message === "string" && message) return message;
  }
  return `Request failed (${status})`;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  return body as T;
}

export function isProviderKeyRotationError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const normalized = error.message.toLowerCase();
  return (
    normalized.includes("public key may have changed") ||
    normalized.includes("server key may have changed")
  );
}

let cachedPublicKeyPem: string | null = null;
let cachedForgeKey: forge.pki.rsa.PublicKey | null = null;

export function clearProviderPublicKeyCache(): void {
  cachedPublicKeyPem = null;
  cachedForgeKey = null;
}

async function importProviderPublicKey(
  forceRefresh = false,
): Promise<forge.pki.rsa.PublicKey> {
  if (!forceRefresh && cachedForgeKey) {
    return cachedForgeKey;
  }
  const response = await authFetch("/api/providers/public-key");
  const body = await parseJsonOrThrow<{ public_key: string }>(response);
  const publicKeyPem = body.public_key?.trim();
  if (!publicKeyPem) {
    throw new Error("Provider public key is missing.");
  }
  if (!forceRefresh && cachedPublicKeyPem === publicKeyPem && cachedForgeKey) {
    return cachedForgeKey;
  }
  const forgeKey = forge.pki.publicKeyFromPem(publicKeyPem);
  cachedPublicKeyPem = publicKeyPem;
  cachedForgeKey = forgeKey;
  return forgeKey;
}

const ENVELOPE_VERSION = "v1";
const ENVELOPE_AAD = "unsloth-studio-provider-key-v1";
const AES_KEY_BYTES = 32;
const NONCE_BYTES = 12;

function randomBinaryString(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return String.fromCharCode(...bytes);
}

/** RSA-OAEP wraps only the 32-byte content key: wrapping the API key itself caps it at 190 bytes. */
export async function encryptProviderApiKey(
  plaintextApiKey: string,
  forceRefresh = false,
): Promise<string> {
  const key = await importProviderPublicKey(forceRefresh);
  const aesKey = randomBinaryString(AES_KEY_BYTES);
  const nonce = randomBinaryString(NONCE_BYTES);

  const cipher = forge.cipher.createCipher("AES-GCM", aesKey);
  cipher.start({ iv: nonce, additionalData: ENVELOPE_AAD, tagLength: 128 });
  // forge ciphers take bytes: without encodeUtf8 a non-ASCII key loses all but each low byte.
  cipher.update(forge.util.createBuffer(forge.util.encodeUtf8(plaintextApiKey)));
  if (!cipher.finish()) {
    throw new Error("Failed to encrypt API key.");
  }

  const wrappedKey = key.encrypt(aesKey, "RSA-OAEP", {
    md: forge.md.sha256.create(),
    mgf1: { md: forge.md.sha256.create() },
  });
  return [
    ENVELOPE_VERSION,
    forge.util.encode64(wrappedKey),
    forge.util.encode64(nonce),
    forge.util.encode64(cipher.output.getBytes() + cipher.mode.tag.getBytes()),
  ].join(".");
}

export async function listProviderRegistry(): Promise<ProviderRegistryEntry[]> {
  // include_hidden asks for the backend-only entries (the self-hosted presets), which carry the
  // studio-tools capability the composer gates on. An older backend ignores the parameter and
  // returns the visible entries, so the capability reads as unknown and the pills stay closed.
  const response = await authFetch(
    "/api/providers/registry?include_hidden=true&include_oauth=true",
  );
  return parseJsonOrThrow<ProviderRegistryEntry[]>(response);
}

export async function listProviderConfigs(): Promise<ProviderConfig[]> {
  const response = await authFetch("/api/providers/");
  return parseJsonOrThrow<ProviderConfig[]>(response);
}

export async function createProviderConfig(payload: {
  providerType: string;
  displayName: string;
  baseUrl?: string | null;
  apiType?: ConnectionApiType;
  models?: string[];
  availableModels?: string[];
  maxOutputTokens?: number | null;
  capabilities?: ProviderCapabilities | null;
  managedConfig?: KaggleTPUManagedConfig | null;
  reasoningConfig?: CustomReasoningConfig | null;
  apiKey?: string;
  kaggleApiToken?: string;
}): Promise<ProviderConfig> {
  return withApiKeyEncryptionRetry(
    payload.apiKey ?? "",
    async (encryptedApiKey) => {
      return withApiKeyEncryptionRetry(
        payload.kaggleApiToken ?? "",
        async (encryptedKaggleApiToken) => {
          const response = await authFetch("/api/providers/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              provider_type: payload.providerType,
              display_name: payload.displayName,
              api_type: payload.apiType ?? "chat_completions",
              ...(payload.reasoningConfig === undefined ? {} : {reasoning_config: payload.reasoningConfig}),
              base_url: payload.baseUrl ?? null,
              models: payload.models ?? [],
              available_models: payload.availableModels ?? [],
              ...(payload.maxOutputTokens === undefined
                ? {}
                : { max_output_tokens: payload.maxOutputTokens }),
              ...(payload.capabilities === undefined
                ? {}
                : { capabilities: payload.capabilities }),
              ...(payload.managedConfig === undefined
                ? {}
                : { managed_config: payload.managedConfig }),
              encrypted_api_key: encryptedApiKey,
              encrypted_kaggle_api_token: encryptedKaggleApiToken,
            }),
          });
          return parseJsonOrThrow<ProviderConfig>(response);
        },
      );
    },
  );
}

export async function deleteProviderConfig(providerId: string): Promise<void> {
  const response = await authFetch(`/api/providers/${providerId}`, {
    method: "DELETE",
  });
  // Treat 404 as success: another tab already deleted this provider, so pruning the stale cache is
  // correct. Otherwise the caller throws and the user is stuck with an entry they cannot remove.
  if (response.status === 404) {
    return;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
}

export async function updateProviderConfig(
  providerId: string,
  payload: {
    displayName?: string;
    baseUrl?: string | null;
    apiType?: ConnectionApiType;
    isEnabled?: boolean;
    models?: string[];
    availableModels?: string[];
    maxOutputTokens?: number | null;
    capabilities?: ProviderCapabilities | null;
    managedConfig?: KaggleTPUManagedConfig | null;
    reasoningConfig?: CustomReasoningConfig | null;
    apiKey?: string;
    clearApiKey?: boolean;
    kaggleApiToken?: string;
    clearKaggleApiToken?: boolean;
  },
): Promise<ProviderConfig> {
  return withApiKeyEncryptionRetry(
    payload.apiKey ?? "",
    async (encryptedApiKey) => {
      return withApiKeyEncryptionRetry(
        payload.kaggleApiToken ?? "",
        async (encryptedKaggleApiToken) => {
          const response = await authFetch(`/api/providers/${providerId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...(payload.apiType === undefined ? {} : {api_type: payload.apiType}),
              ...(payload.reasoningConfig === undefined ? {} : {reasoning_config: payload.reasoningConfig}),
              ...(payload.displayName === undefined
                ? {}
                : { display_name: payload.displayName }),
              ...(payload.baseUrl === undefined
                ? {}
                : { base_url: payload.baseUrl }),
              ...(payload.isEnabled === undefined
                ? {}
                : { is_enabled: payload.isEnabled }),
              ...(payload.models === undefined ? {} : { models: payload.models }),
              ...(payload.availableModels === undefined
                ? {}
                : { available_models: payload.availableModels }),
              ...(payload.maxOutputTokens === undefined
                ? {}
                : { max_output_tokens: payload.maxOutputTokens }),
              ...(payload.capabilities === undefined
                ? {}
                : { capabilities: payload.capabilities }),
              ...(payload.managedConfig === undefined
                ? {}
                : { managed_config: payload.managedConfig }),
              ...(payload.apiKey === undefined
                ? {}
                : { encrypted_api_key: encryptedApiKey }),
              ...(payload.clearApiKey === undefined
                ? {}
                : { clear_api_key: payload.clearApiKey }),
              ...(payload.kaggleApiToken === undefined
                ? {}
                : { encrypted_kaggle_api_token: encryptedKaggleApiToken }),
              ...(payload.clearKaggleApiToken === undefined
                ? {}
                : { clear_kaggle_api_token: payload.clearKaggleApiToken }),
            }),
          });
          return parseJsonOrThrow<ProviderConfig>(response);
        },
      );
    },
  );
}

export async function migrateProviderApiKey(
  providerId: string,
  apiKey: string,
): Promise<ProviderConfig> {
  return withApiKeyEncryptionRetry(apiKey, async (encryptedApiKey) => {
    const response = await authFetch(`/api/providers/${providerId}/api-key/migrate`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ encrypted_api_key: encryptedApiKey }),
    });
    return parseJsonOrThrow<ProviderConfig>(response);
  });
}


async function withApiKeyEncryptionRetry<T>(
  plaintextApiKey: string,
  call: (encryptedApiKey: string | null) => Promise<T>,
): Promise<T> {
  // Empty key (local providers): skip RSA round-trip and let the backend omit auth.
  if (!plaintextApiKey) {
    return await call(null);
  }
  try {
    const encrypted = await encryptProviderApiKey(plaintextApiKey, false);
    return await call(encrypted);
  } catch (error) {
    if (!isProviderKeyRotationError(error)) {
      throw error;
    }
    clearProviderPublicKeyCache();
    const encrypted = await encryptProviderApiKey(plaintextApiKey, true);
    return await call(encrypted);
  }
}

export async function testProviderConnection(payload: {
  providerType: string;

  providerId?: string | null;
  apiKey: string;
  baseUrl?: string | null;
  apiType?: ConnectionApiType;
  modelId?: string | null;
}): Promise<ProviderTestResult> {
  return withApiKeyEncryptionRetry(payload.apiKey, async (encryptedApiKey) => {
    const response = await authFetch("/api/providers/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_type: payload.providerType,

        provider_id: payload.providerId ?? null,
        encrypted_api_key: encryptedApiKey,
        base_url: payload.baseUrl ?? null,
        ...(payload.apiType === undefined ? {} : { api_type: payload.apiType }),
        model_id: payload.modelId ?? null,
      }),
    });
    return parseJsonOrThrow<ProviderTestResult>(response);
  });
}

export async function listProviderModels(payload: {
  providerType: string;

  providerId?: string | null;
  apiKey: string;
  baseUrl?: string | null;
  apiType?: ConnectionApiType;
}): Promise<ProviderModelInfo[]> {
  return withApiKeyEncryptionRetry(payload.apiKey, async (encryptedApiKey) => {
    const response = await authFetch("/api/providers/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_type: payload.providerType,

        provider_id: payload.providerId ?? null,
        encrypted_api_key: encryptedApiKey,
        base_url: payload.baseUrl ?? null,
        ...(payload.apiType === undefined ? {} : { api_type: payload.apiType }),
      }),
    });
    return parseJsonOrThrow<ProviderModelInfo[]>(response);
  });
}

async function updateKaggleLifecycle(
  providerId: string,
  action: "start" | "reconnect" | "stop",
): Promise<KaggleTPULifecycleStatus> {
  const response = await authFetch(
    `/api/providers/${providerId}/kaggle/${action}`,
    { method: "POST" },
  );
  return parseJsonOrThrow<KaggleTPULifecycleStatus>(response);
}

export async function getKaggleTPUStatus(
  providerId: string,
): Promise<KaggleTPULifecycleStatus> {
  const response = await authFetch(
    `/api/providers/${providerId}/kaggle/status`,
  );
  return parseJsonOrThrow<KaggleTPULifecycleStatus>(response);
}

export const startKaggleTPU = (providerId: string) =>
  updateKaggleLifecycle(providerId, "start");

export const reconnectKaggleTPU = (providerId: string) =>
  updateKaggleLifecycle(providerId, "reconnect");

export const stopKaggleTPU = (providerId: string) =>
  updateKaggleLifecycle(providerId, "stop");
export async function fetchModelCatalog(): Promise<ModelCatalogResponse> {
  const response = await authFetch("/api/providers/model-catalog");
  return parseJsonOrThrow<ModelCatalogResponse>(response);
}

export async function listProviderModelCapabilities(payload: {
  providerType: string;
  providerId?: string | null;
  apiKey: string;
  baseUrl?: string | null;
  apiType?: ProviderApiType;
}): Promise<ProviderModelCapabilityInfo[]> {
  return withApiKeyEncryptionRetry(payload.apiKey, async (encryptedApiKey) => {
    const response = await authFetch("/api/providers/model-capabilities", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_type: payload.providerType,
        provider_id: payload.providerId ?? null,
        encrypted_api_key: encryptedApiKey,
        base_url: payload.baseUrl ?? null,
        ...(payload.apiType === undefined ? {} : { api_type: payload.apiType }),
      }),
    });
    return parseJsonOrThrow<ProviderModelCapabilityInfo[]>(response);
  });
}


export interface CodexSubscriptionModels {
  models: ProviderModelInfo[];
  /** Every model the plan returned, offered or not: absent from this means the account cannot reach
   *  it, while present-but-unoffered only means it is no longer shown. */
  known?: ProviderModelInfo[];
  /** "reauthorization_required" is a curated answer that also says the connection has to be
   *  reconnected: the picker must not treat it as the plan's catalog. */
  source: "subscription" | "curated" | "reauthorization_required";
}

export async function fetchCodexSubscriptionModels(
  providerId: string,
  options?: { refresh?: boolean },
): Promise<CodexSubscriptionModels> {
  // An explicit reload asks about plan changes, so it must not be served from cache.
  const query = options?.refresh ? "?refresh=true" : "";
  const response = await authFetch(`/api/providers/${providerId}/codex/models${query}`);
  return parseJsonOrThrow<CodexSubscriptionModels>(response);
}


export interface CodexOAuthFlow {
  flow_id: string;
  method: "browser" | "device";
  status: "pending" | "connected" | "error" | "cancelled";
  expires_at: number;
  authorization_url?: string | null;
  verification_url?: string | null;
  user_code?: string | null;
  message?: string | null;
}

export async function startCodexOAuth(
  providerId: string,
  method: "browser" | "device",
): Promise<CodexOAuthFlow> {
  const response = await authFetch(`/api/providers/${providerId}/oauth/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method }),
  });
  return parseJsonOrThrow<CodexOAuthFlow>(response);
}

export async function getCodexOAuthFlow(providerId: string, flowId: string): Promise<CodexOAuthFlow> {
  const response = await authFetch(`/api/providers/${providerId}/oauth/flows/${flowId}`);
  return parseJsonOrThrow<CodexOAuthFlow>(response);
}

export async function completeCodexOAuth(
  providerId: string,
  flowId: string,
  callbackUrl: string,
): Promise<CodexOAuthFlow> {
  const response = await authFetch(`/api/providers/${providerId}/oauth/flows/${flowId}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ callback_url: callbackUrl }),
  });
  return parseJsonOrThrow<CodexOAuthFlow>(response);
}

export async function cancelCodexOAuthFlow(
  providerId: string,
  flowId: string,
): Promise<void> {
  const response = await authFetch(
    `/api/providers/${providerId}/oauth/flows/${flowId}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
}



export async function disconnectCodexOAuth(providerId: string): Promise<void> {
  const response = await authFetch(`/api/providers/${providerId}/oauth`, { method: "DELETE" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
}
