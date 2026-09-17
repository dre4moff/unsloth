// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowLeft02Icon,
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
  Wifi02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Eye, EyeOff } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ApiProviderLogo } from "./api-provider-logo";

import { OpenAICodexConnect } from "./openai-codex-connect";
import {
  type KaggleTPULifecycleStatus,
  type ProviderRegistryEntry,
  conservativeDiscoveredContextLength,
  createProviderConfig,
  deleteProviderConfig,
  getKaggleTPUStatus,
  listProviderModels,
  listProviderRegistry,
  reconnectKaggleTPU,
  startKaggleTPU,
  stopKaggleTPU,
  testProviderConnection,
  updateProviderConfig,
} from "./api/providers-api";

import { resolveProviderCredentialEdit } from "./provider-credential-edit";
import { getExternalMinOutputTokens } from "./provider-capabilities";
import type { ExternalProviderConfig } from "./external-providers";
import {
  CUSTOM_PROVIDER_PRESETS,
  allowsManualModelIdsWithCatalog,
  customProviderBaseUrlPlaceholder,
  customProviderDisplayName,
  customProviderModelIdsPlaceholder,
  customPresetSkipsApiKeyField,
  PROVIDER_MAX_OUTPUT_TOKENS_MIN,
  getExternalProviderApiKey,
  isCustomProviderType,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  CUSTOM_PROVIDER_DISPLAY_NAME,
  providerModelSupportsStudioTools,
  removeExternalProviderApiKey,
  supportsProviderMaxOutputTokens,
  supportsProviderReasoningToggle,
  supportsRemoteModelCatalog,
  toExternalBackendProviderType,
} from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";
import {
  pruneProviderModelIds,
  syncExternalProvidersFromBackend,
} from "./sync-external-providers";

/** Matches navbar / thread layout easing (see index.css --ease-out-quart) */
const PROVIDER_FORM_EASE: [number, number, number, number] = [
  0.165, 0.84, 0.44, 1,
];
const PROVIDER_FORM_DURATION = 0.2;
const CUSTOM_PROVIDER_MISSING_KEY_MESSAGE =
  "No API key found. Add a valid API key for this connection.";
const HIDDEN_PROVIDER_TYPES = new Set(["qwen"]);

function parseManualModelIds(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of text.split(/[\n,]+/)) {
    const id = raw.trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

// Remote providers that support both catalog load and manual model IDs.
const EMPTY_CATALOG_HINTS: Record<string, { title: string; description: string }> =
  {
    ollama: {
      title: "No local Ollama models found.",
      description:
        "Run `ollama pull <model>` in a terminal, then reload — or enter a model ID manually below.",
    },
    llama_cpp: {
      title: "No llama.cpp models found.",
      description:
        "Ensure llama-server is running with models loaded, then reload — or enter model IDs manually below.",
    },
    vllm: {
      title: "No vLLM models found.",
      description:
        "Ensure the vLLM server is running and models are loaded, then reload — or enter model IDs manually below.",
    },
  };

function emptyCatalogHint(providerType: string): {
  title: string;
  description: string;
} {
  return (
    EMPTY_CATALOG_HINTS[providerType] ?? {
      title: "No models returned by this connection.",
      description: "Enter model IDs manually below, or check the server.",
    }
  );
}

function shouldAppendOpenAiVersionPath(providerType: string): boolean {
  return (
    providerType === "ollama" ||
    providerType === "llama_cpp" ||
    providerType === "vllm" ||
    providerType === "openai_compatible" ||
    providerType === "kaggle_tpu" ||
    providerType === LEGACY_CUSTOM_PROVIDER_TYPE
  );
}

function formatModelSummary(models: string[]): string {
  if (models.length === 0) {
    return "No models enabled";
  }
  const visible = models.slice(0, 4);
  const remaining = models.length - visible.length;
  return `${visible.join(", ")}${remaining > 0 ? ` +${remaining}` : ""}`;
}

interface ChatProvidersSettingsProps {
  providers: ExternalProviderConfig[];
  onProvidersChange: (providers: ExternalProviderConfig[]) => void;
}

export function ChatProvidersSettings({
  providers,
  onProvidersChange,
}: ChatProvidersSettingsProps) {
  const providersRef = useRef(providers);
  const seededProviderTypeRef = useRef<string | null>(null);
  // Latches the one-shot auto-open below. Every navigation the user drives sets
  // it too, so a slow first sync cannot pull them back into the form.
  const autoOpenedAddFormRef = useRef(false);
  const [page, setPage] = useState<"list" | "form">("list");
  const [providerType, setProviderType] = useState<string>("");
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);

  const [clearApiKeyRequested, setClearApiKeyRequested] = useState(false);
  const [baseUrlDraft, setBaseUrlDraft] = useState("");
  const [maxOutputTokensDraft, setMaxOutputTokensDraft] = useState("");
  const [editingBackendProviderType, setEditingBackendProviderType] = useState<
    string | null
  >(null);
  const [editingProviderId, setEditingProviderId] = useState<string | null>(
    null,
  );
  const [registry, setRegistry] = useState<ProviderRegistryEntry[]>([]);
  const [registryMetadata, setRegistryMetadata] = useState<
    ProviderRegistryEntry[]
  >([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [syncingProviders, setSyncingProviders] = useState(false);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [mutatingProvider, setMutatingProvider] = useState(false);
  const [manualModelIds, setManualModelIds] = useState("");
  const [modelSearchQuery, setModelSearchQuery] = useState("");
  const [customProviderName, setCustomProviderName] = useState(
    CUSTOM_PROVIDER_DISPLAY_NAME,
  );
  const [isReasoningModel, setIsReasoningModel] = useState(false);
  const [supportsToolCalling, setSupportsToolCalling] = useState(true);
  const [supportsVision, setSupportsVision] = useState(false);
  const [contextLengthDraft, setContextLengthDraft] = useState("");
  const [kaggleApiToken, setKaggleApiToken] = useState("");
  const [showKaggleApiToken, setShowKaggleApiToken] = useState(false);
  const [clearKaggleApiTokenRequested, setClearKaggleApiTokenRequested] =
    useState(false);
  const [kaggleAutoStart, setKaggleAutoStart] = useState(false);
  const [kaggleAutoStop, setKaggleAutoStop] = useState(false);
  const [kaggleTextOnly, setKaggleTextOnly] = useState(false);
  const [kaggleFastStart, setKaggleFastStart] = useState(false);
  const [kaggleKeepaliveDraft, setKaggleKeepaliveDraft] = useState("480");
  const [kaggleStatus, setKaggleStatus] =
    useState<KaggleTPULifecycleStatus | null>(null);
  const [kaggleActionPending, setKaggleActionPending] = useState(false);
  const reduceMotion = useReducedMotion();
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const setConnectionsEnabled = useExternalProvidersStore(
    (s) => s.setConnectionsEnabled,
  );
  const isCustomProvider = isCustomProviderType(providerType);
  const isOpenAICompatible = providerType === "openai_compatible";
  const isKaggleTPU = providerType === "kaggle_tpu";
  const showsConnectionCapabilities = isOpenAICompatible || isKaggleTPU;
  // a connection being created has no stored type yet, so only the UI type can decide
  const supportsMaxOutputTokens = supportsProviderMaxOutputTokens(
    providerType,
    editingProviderId ? editingBackendProviderType : null,
  );
  // llama.cpp hides the key field. Ollama and vLLM show an optional key:
  // Ollama cloud and secured vLLM need one; local servers leave it empty.
  const showReasoningToggle = supportsProviderReasoningToggle(providerType);
  // Studio runs Search, Code, MCP and RAG on this machine for any provider that
  // advertises the capability, with no extra opt-in. Say so where the
  // connection is created: the tool results also travel back to the provider as
  // the next turn's input, which is not obvious from "connect a model".
  const runsStudioToolsLocally =
    providerModelSupportsStudioTools(
      toExternalBackendProviderType(providerType),
      null,
    ) === true;

  const registryByType = useMemo(
    () =>
      new Map(registryMetadata.map((entry) => [entry.provider_type, entry])),
    [registryMetadata],
  );
  const selectedProviderContract = registryByType.get(
    toExternalBackendProviderType(providerType),
  );
  const usesOAuth = selectedProviderContract?.auth_kind === "chatgpt_oauth";

  const isCodexSubscription = usesOAuth;
  const modelIdsEditable =
    selectedProviderContract?.model_ids_editable !== false;
  const showApiKeyField =
    !usesOAuth && !isKaggleTPU && !customPresetSkipsApiKeyField(providerType);
  const isCuratedModelList = useMemo(() => {
    return registryByType.get(providerType)?.model_list_mode === "curated";
  }, [registryByType, providerType]);
  const isManualModelList =
    (isCustomProvider && !supportsRemoteModelCatalog(providerType)) ||
    (isCuratedModelList && modelIdsEditable);

  const modelsPanelKey = isCustomProvider
    ? providerType || "custom"
    : isCuratedModelList
      ? "curated"
      : "remote";
  const remoteAllowsManual = allowsManualModelIdsWithCatalog(providerType);
  const formModelCount =
    isManualModelList || remoteAllowsManual
      ? new Set([...selectedModelIds, ...parseManualModelIds(manualModelIds)])
          .size
      : selectedModelIds.length;
  const modelStatusLabel =
    !isManualModelList &&
    !remoteAllowsManual &&
    availableModels.length === 0
      ? "No models loaded"
      : `${formModelCount} ${formModelCount === 1 ? "model" : "models"} selected`;
  const showModelsBody =
    isManualModelList ||
    remoteAllowsManual ||
    availableModels.length > 0;
  const missingModelCatalogBaseUrl =
    supportsRemoteModelCatalog(providerType) && baseUrlDraft.trim().length === 0;
  const editingProviderHasSavedKey = Boolean(
    editingProviderId &&
      providers.find((provider) => provider.id === editingProviderId)?.hasApiKey,
  );
  const editingProviderHasSavedKaggleToken = Boolean(
    editingProviderId &&
      providers.find((provider) => provider.id === editingProviderId)
        ?.hasKaggleApiToken,
  );
  const missingModelCatalogApiKey =
    !isCustomProvider &&
    !isCuratedModelList &&
    apiKey.trim().length === 0 &&
    !(editingProviderHasSavedKey && !clearApiKeyRequested);
  const loadModelsDisabled =
    modelsLoading ||
    mutatingProvider ||
    isManualModelList ||
    missingModelCatalogBaseUrl ||
    missingModelCatalogApiKey;
  const loadModelsTitle =
    isManualModelList && isCustomProvider
      ? "This connection uses manual model IDs"
      : isCuratedModelList
        ? "Full catalog is not fetched for this connection"
        : missingModelCatalogBaseUrl
          ? "Enter a Base URL before loading models"
          : missingModelCatalogApiKey
            ? "Enter an API key before loading models"
            : undefined;
  const filteredAvailableModels = useMemo(() => {
    const query = modelSearchQuery.trim().toLowerCase();
    if (!query) {
      return availableModels;
    }
    return availableModels.filter((model) =>
      model.toLowerCase().includes(query),
    );
  }, [availableModels, modelSearchQuery]);
  const availableModelsLabel = modelSearchQuery.trim()
    ? `${filteredAvailableModels.length} of ${availableModels.length} models`
    : `${availableModels.length} models`;
  const modelSearchInputClassName =
    "h-8 w-full bg-background/55 text-xs placeholder:text-muted-foreground/65 focus-visible:border-border focus-visible:ring-0";

  useEffect(() => {
    providersRef.current = providers;
  }, [providers]);

  useEffect(() => {
    if (!providerType || editingProviderId) return;
    if (seededProviderTypeRef.current === providerType) return;
    seededProviderTypeRef.current = providerType;
    setMaxOutputTokensDraft("");
    setEditingBackendProviderType(null);
    const entry = registryByType.get(providerType);
    if (!entry) {
      if (isCustomProviderType(providerType)) {
        setCustomProviderName(customProviderDisplayName(providerType));
        setBaseUrlDraft("");
      }
      return;
    }
    // Seed default_models only for curated providers (catalog too large to
    // enumerate). Remote cloud providers and local OpenAI-compat presets stay
    // empty until the user clicks "Load available models".
    const seedDefaults = entry.model_list_mode === "curated";
    setAvailableModels(seedDefaults ? [...entry.default_models] : []);
    setSelectedModelIds([]);
    setManualModelIds(
      providerType === "kaggle_tpu" ? entry.default_models.join("\n") : "",
    );
    setModelSearchQuery("");
    setBaseUrlDraft("");
    setSupportsToolCalling(entry.supports_tool_calling !== false);
    setIsReasoningModel(entry.supports_reasoning === true);
    setSupportsVision(entry.supports_vision === true);
    setContextLengthDraft(
      typeof entry.context_length === "number"
        ? entry.context_length.toString()
        : "",
    );
  }, [providerType, editingProviderId, registryByType]);

  const totalModels = useMemo(
    () =>
      providers.reduce((count, provider) => count + provider.models.length, 0),
    [providers],
  );

  useEffect(() => {
    let isMounted = true;
    const syncFromBackend = async ({
      showSpinner = true,
    }: { showSpinner?: boolean } = {}) => {
      if (showSpinner) {
        setRegistryLoading(true);
        setSyncingProviders(true);
      }
      let syncSucceeded = false;
      try {
        const [registryRows, syncedProviders] = await Promise.all([
          listProviderRegistry(),
          syncExternalProvidersFromBackend(providersRef.current),
        ]);
        if (!isMounted) return;
        syncSucceeded = true;
        // Hidden entries are fetched for their capabilities only; the dropdown
        // surfaces them through CUSTOM_PROVIDER_PRESETS instead.
        const selectableRegistry = registryRows.filter((entry) => !entry.hidden);
        setRegistry(selectableRegistry);
        setRegistryMetadata(registryRows);
        setProviderType((current) => {
          if (
            current &&
            (isCustomProviderType(current) ||
              registryRows.some((entry) => entry.provider_type === current))
          ) {
            return current;
          }
          return registryRows[0]?.provider_type ?? "";
        });
        // Trust the backend response. An empty array means every connection was
        // removed (often from another tab); mirror that locally, else stale
        // entries become un-removable here until localStorage is cleared.
        onProvidersChange(syncedProviders);
        // An empty list never says what this page is for, so open the form
        // instead. Reads the synced response, not the local snapshot, so a
        // stale empty list cannot flash the form at an existing user. Once
        // only, else the focus re-sync would pull the user back here.
        if (!autoOpenedAddFormRef.current) {
          autoOpenedAddFormRef.current = true;
          if (syncedProviders.length === 0 && selectableRegistry.length > 0) {
            setPage("form");
          }
        }
      } catch (error) {
        // Only surface a toast for real failures, not for the silent
        // background re-sync on tab focus.
        if (showSpinner) {
          const message =
            error instanceof Error ? error.message : "Unknown error";
          toast.error(`Failed to load connections: ${message}`);
        }
      } finally {
        if (isMounted && showSpinner) {
          setRegistryLoading(false);
          setSyncingProviders(false);
        }
      }
      return syncSucceeded;
    };
    void syncFromBackend();
    // Re-sync silently when the tab regains focus so deletes made in
    // another browser propagate without forcing the user to reopen the
    // dialog. Skip when the document is hidden to avoid background work.
    const handleVisibilityChange = () => {
      if (typeof document === "undefined" || document.hidden) return;
      void syncFromBackend({ showSpinner: false });
    };
    if (typeof window !== "undefined") {
      window.addEventListener("focus", handleVisibilityChange);
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    return () => {
      isMounted = false;
      if (typeof window !== "undefined") {
        window.removeEventListener("focus", handleVisibilityChange);
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
    };
  }, [onProvidersChange]);

  useEffect(() => {
    if (
      page !== "form" ||
      !editingProviderId ||
      providerType !== "kaggle_tpu"
    ) {
      setKaggleStatus(null);
      return;
    }
    let active = true;
    const refreshStatus = async () => {
      try {
        const status = await getKaggleTPUStatus(editingProviderId);
        if (!active) return;
        setKaggleStatus(status);
        if (status.state === "READY" && status.base_url) {
          setBaseUrlDraft(status.base_url);
          if (status.model) setManualModelIds(status.model);
          if (status.context_length) {
            setContextLengthDraft(status.context_length.toString());
          }
          onProvidersChange(
            providersRef.current.map((provider) =>
              provider.id === editingProviderId
                ? {
                    ...provider,
                    baseUrl: status.base_url ?? provider.baseUrl,
                    models: status.model ? [status.model] : provider.models,
                    availableModels: status.model
                      ? [status.model]
                      : provider.availableModels,
                    capabilities: {
                      ...(provider.capabilities ?? {
                        supports_streaming: true,
                        supports_tool_calling: true,
                        supports_reasoning: true,
                        supports_vision: true,
                        supports_images: true,
                        api_mode: "chat_completions" as const,
                      }),
                      context_length:
                        status.context_length ??
                        provider.capabilities?.context_length,
                    },
                  }
                : provider,
            ),
          );
        }
      } catch {
        if (active) setKaggleStatus(null);
      }
    };
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [editingProviderId, onProvidersChange, page, providerType]);

  async function runKaggleAction(action: "start" | "reconnect" | "stop") {
    if (!editingProviderId) return;
    setKaggleActionPending(true);
    try {
      const status =
        action === "start"
          ? await startKaggleTPU(editingProviderId)
          : action === "reconnect"
            ? await reconnectKaggleTPU(editingProviderId)
            : await stopKaggleTPU(editingProviderId);
      setKaggleStatus(status);
      if (status.state === "ERROR" || status.state === "DISCONNECTED") {
        toast.error(status.message || `Kaggle TPU: ${status.state}`);
      } else {
        toast.success(status.message || `Kaggle TPU: ${status.state}`);
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Kaggle TPU action failed.",
      );
    } finally {
      setKaggleActionPending(false);
    }
  }

  function resetForm() {
    setEditingProviderId(null);
    setApiKey("");

    setClearApiKeyRequested(false);
    setShowApiKey(false);
    setBaseUrlDraft("");
    setMaxOutputTokensDraft("");
    setEditingBackendProviderType(null);
    setAvailableModels([]);
    setSelectedModelIds([]);
    setManualModelIds("");
    setModelSearchQuery("");
    setCustomProviderName(customProviderDisplayName(providerType));
    setIsReasoningModel(false);
    setSupportsToolCalling(true);
    setSupportsVision(false);
    setContextLengthDraft("");
    setKaggleApiToken("");
    setShowKaggleApiToken(false);
    setClearKaggleApiTokenRequested(false);
    setKaggleAutoStart(false);
    setKaggleAutoStop(false);
    setKaggleTextOnly(false);
    setKaggleFastStart(false);
    setKaggleKeepaliveDraft("480");
    setKaggleStatus(null);
  }

  function openAddProvider() {
    resetForm();
    const entry = providerType ? registryByType.get(providerType) : null;
    if (entry?.model_list_mode === "curated") {
      setAvailableModels([...entry.default_models]);
    }
    if (entry) {
      setSupportsToolCalling(entry.supports_tool_calling !== false);
      setIsReasoningModel(entry.supports_reasoning === true);
      setSupportsVision(entry.supports_vision === true);
      setContextLengthDraft(
        typeof entry.context_length === "number"
          ? entry.context_length.toString()
          : "",
      );
      if (providerType === "kaggle_tpu") {
        setManualModelIds(entry.default_models.join("\n"));
      }
    }
    autoOpenedAddFormRef.current = true;
    setPage("form");
  }

  function closeForm() {
    resetForm();
    autoOpenedAddFormRef.current = true;
    setPage("list");
  }

  function toggleModel(modelId: string) {
    setSelectedModelIds((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId],
    );
  }

  function selectAllModels() {
    setSelectedModelIds([...availableModels]);
  }

  function clearModelSelection() {
    setSelectedModelIds([]);
  }

  function parseOptionalBaseUrl(
    input: string,
    options: { appendOpenAiVersionPath?: boolean } = {},
  ): string | null {
    const trimmed = input.trim();
    if (!trimmed) return null;
    let parsed: URL;
    try {
      parsed = new URL(trimmed);
    } catch {
      throw new Error("Base URL must be a valid URL.");
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Base URL must use http or https.");
    }
    if (options.appendOpenAiVersionPath) {
      const pathname = parsed.pathname.replace(/\/+$/, "");
      if (!pathname) {
        parsed.pathname = "/v1";
      }
    }
    return parsed.toString().replace(/\/+$/, "");
  }

  function parseBaseUrlForProvider(
    input: string,
    required: boolean,
    providerTypeForUrl: string,
  ): string | null {
    const trimmed = input.trim();
    if (!trimmed) {
      if (required) {
        throw new Error("Base URL is required for this connection.");
      }
      return null;
    }
    return parseOptionalBaseUrl(trimmed, {
      appendOpenAiVersionPath: shouldAppendOpenAiVersionPath(providerTypeForUrl),
    });
  }

  function parseMaxOutputTokens(input: string): number | null {
    const trimmed = input.trim();
    if (!trimmed) return null;
    if (!/^\d+$/.test(trimmed)) {
      throw new Error("Max Tokens limit must be an integer.");
    }
    const value = Number(trimmed);
    if (!Number.isSafeInteger(value)) {
      throw new Error("Max Tokens limit must be a safe integer.");
    }
    // getExternalMaxOutputTokens raises a sub-floor cap anyway, so say so instead of storing it
    const floor = Math.max(
      PROVIDER_MAX_OUTPUT_TOKENS_MIN,
      getExternalMinOutputTokens(providerType),
    );
    if (value < floor) {
      throw new Error(
        `Max Tokens limit must be at least ${floor.toLocaleString()}.`,
      );
    }
    return value;
  }

  function parseIntegerField(
    input: string,
    label: string,
    minimum: number,
    maximum: number,
  ): number {
    const trimmed = input.trim();
    if (!/^\d+$/.test(trimmed)) {
      throw new Error(`${label} must be an integer.`);
    }
    const value = Number(trimmed);
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
      throw new Error(
        `${label} must be between ${minimum.toLocaleString()} and ${maximum.toLocaleString()}.`,
      );
    }
    return value;
  }

  function buildConnectionCapabilities() {
    if (!showsConnectionCapabilities) return undefined;
    const contextLength = contextLengthDraft.trim()
      ? parseIntegerField(
          contextLengthDraft,
          "Context length",
          1024,
          Number.MAX_SAFE_INTEGER,
        )
      : null;
    return {
      supports_streaming: true,
      supports_tool_calling: supportsToolCalling,
      supports_reasoning: isKaggleTPU || isReasoningModel,
      supports_vision: !kaggleTextOnly && supportsVision,
      supports_images: !kaggleTextOnly && supportsVision,
      context_length: contextLength,
      api_mode: "chat_completions" as const,
    };
  }

  function buildKaggleManagedConfig() {
    if (!isKaggleTPU) return undefined;
    return {
      lab_path: null,
      auto_start: kaggleAutoStart,
      auto_stop: kaggleAutoStop,
      keepalive_minutes: parseIntegerField(
        kaggleKeepaliveDraft,
        "Keepalive",
        1,
        540,
      ),
      text_only: kaggleTextOnly,
      fast_start: kaggleFastStart,
      max_num_seqs: 4,
      mtp_tokens: 3,
      reasoning_effort_default: "xhigh" as const,
    };
  }

  async function loadModels() {
    if (!providerType) {
      toast.error("Choose a connection first.");
      return;
    }
    if (isCustomProvider && !supportsRemoteModelCatalog(providerType)) {
      toast.info("This connection uses manual model IDs.");
      return;
    }
    if (isCuratedModelList) {
      toast.info(
        "This connection has a very large model catalog. Use the suggestions and add model IDs manually — full list is not fetched.",
      );
      return;
    }
    if (!isCustomProvider && !apiKey.trim() && !editingProviderHasSavedKey) {
      toast.error("Add an API key first.");
      return;
    }
    setModelsLoading(true);
    try {
      const baseUrl = parseBaseUrlForProvider(
        baseUrlDraft,
        supportsRemoteModelCatalog(providerType),
        providerType,
      );
      const backendProviderType =
        toExternalBackendProviderType(providerType) ?? providerType;
      const models = await listProviderModels({
        providerType: backendProviderType,

        providerId: editingProviderId,
        apiKey: apiKey.trim(),
        baseUrl,
      });
      const discoveredContextLength = conservativeDiscoveredContextLength(models);
      if (!contextLengthDraft.trim() && discoveredContextLength) {
        setContextLengthDraft(discoveredContextLength.toString());
      }
      const registryDefaults = supportsRemoteModelCatalog(providerType)
        ? []
        : (registryByType.get(providerType)?.default_models ?? []);
      // Union of registry defaults + fetched models, defaults first so any
      // curated picks (e.g. claude-haiku-4-5) always show even when the
      // provider's /models endpoint omits them.
      const modelIds = pruneProviderModelIds(providerType, [
        ...new Set(
          [
            ...registryDefaults,
            ...models.map((model) => model.id.trim()),
          ].filter((id) => id.length > 0),
        ),
      ]);
      setAvailableModels(modelIds);
      setSelectedModelIds((prev) =>
        prev.filter((id) => modelIds.includes(id)),
      );
      if (modelIds.length === 0) {
        const hint = emptyCatalogHint(providerType);
        toast.info(hint.title, { description: hint.description });
      } else {
        toast.success(
          `Found ${modelIds.length} ${modelIds.length === 1 ? "model" : "models"}.`,
        );
      }
      if (editingProviderId) {
        onProvidersChange(
          providersRef.current.map((provider) =>
            provider.id === editingProviderId
              ? { ...provider, availableModels: modelIds }
              : provider,
          ),
        );
      }
      setModelSearchQuery("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Could not load models: ${message}`);
    } finally {
      setModelsLoading(false);
    }
  }

  async function ensureCodexProvider(): Promise<string> {
    if (editingProviderId) return editingProviderId;
    const backendProviderType = toExternalBackendProviderType(providerType);
    const entry = registryByType.get(backendProviderType);
    if (entry?.auth_kind !== "chatgpt_oauth") {
      throw new Error("This connection does not support ChatGPT authorization.");
    }

    setMutatingProvider(true);
    try {
      const models = pruneProviderModelIds(
        providerType,
        selectedModelIds.length > 0 ? selectedModelIds : entry.default_models,
      );
      const available = pruneProviderModelIds(providerType, [
        ...new Set([...availableModels, ...entry.default_models]),
      ]);
      const created = await createProviderConfig({
        providerType: backendProviderType,
        displayName: entry.display_name,
        baseUrl: null,
        models,
        availableModels: available,
      });
      const provider: ExternalProviderConfig = {
        id: created.id,
        providerType: created.provider_type,
        name: created.display_name,
        baseUrl: created.base_url ?? "",
        models,
        availableModels: available,
        hasApiKey: created.has_api_key,
        authKind: created.auth_kind,
        authStatus: created.auth_status,
        createdAt: Number.isFinite(Date.parse(created.created_at))
          ? Date.parse(created.created_at)
          : Date.now(),
        updatedAt: Number.isFinite(Date.parse(created.updated_at))
          ? Date.parse(created.updated_at)
          : Date.now(),
      };
      const nextProviders = [
        ...providersRef.current.filter((current) => current.id !== created.id),
        provider,
      ];
      providersRef.current = nextProviders;
      onProvidersChange(nextProviders);
      setSelectedModelIds(models);
      setAvailableModels(available);
      setEditingProviderId(created.id);
      return created.id;
    } finally {
      setMutatingProvider(false);
    }
  }


  async function addProvider() {
    if (!providerType) {
      toast.error("Choose a connection first.");
      return;
    }
    const backendProviderType = toExternalBackendProviderType(providerType);
    const selectedRegistryEntry = registryByType.get(backendProviderType);
    const displayName = isCustomProvider
      ? customProviderName.trim() || customProviderDisplayName(providerType)
      : (selectedRegistryEntry?.display_name ?? providerType);
    if (
      !isCustomProvider &&
      !isKaggleTPU &&
      selectedRegistryEntry?.auth_kind !== "chatgpt_oauth" &&
      !apiKey.trim()
    ) {
      toast.error("API key is required.");
      return;
    }
    if (isKaggleTPU && !kaggleApiToken.trim()) {
      toast.error("Kaggle API token is required.");
      return;
    }
    const curated = selectedRegistryEntry?.model_list_mode === "curated";
    const manualOnly =
      (isCustomProvider && !supportsRemoteModelCatalog(providerType)) ||
      (curated && selectedRegistryEntry?.model_ids_editable !== false);
    const remoteAllowsManual = allowsManualModelIdsWithCatalog(providerType);
    const manualIds = parseManualModelIds(manualModelIds);
    const allowManual = manualOnly || remoteAllowsManual;
    const modelsToSave = pruneProviderModelIds(
      providerType,
      allowManual
      ? [
          ...new Set([
            ...selectedModelIds,
            ...manualIds,
          ]),
        ]
      : [...selectedModelIds],
    );
    if (manualOnly) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else if (remoteAllowsManual) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else {
      if (availableModels.length === 0) {
        toast.error(
          "Load available models first, then choose which to enable.",
        );
        return;
      }
      if (selectedModelIds.length === 0) {
        toast.error("Select at least one model.");
        return;
      }
    }
    setMutatingProvider(true);
    try {
      const baseUrl = parseBaseUrlForProvider(
        baseUrlDraft,
        isCustomProvider && !isKaggleTPU,
        providerType,
      );
      const maxOutputTokens = supportsMaxOutputTokens
        ? parseMaxOutputTokens(maxOutputTokensDraft)
        : undefined;
      const created = await createProviderConfig({
        providerType: backendProviderType,
        displayName,
        baseUrl,
        models: modelsToSave,
        availableModels: manualOnly
          ? []
          : pruneProviderModelIds(providerType, availableModels),
        maxOutputTokens,
        capabilities: buildConnectionCapabilities(),
        managedConfig: buildKaggleManagedConfig(),
        apiKey: apiKey.trim(),
        kaggleApiToken: isKaggleTPU ? kaggleApiToken.trim() : undefined,

      });
      const createdAt = Number.isFinite(Date.parse(created.created_at))
        ? Date.parse(created.created_at)
        : Date.now();
      const updatedAt = Number.isFinite(Date.parse(created.updated_at))
        ? Date.parse(created.updated_at)
        : Date.now();
      const uiProviderType = isCustomProvider
        ? providerType
        : created.provider_type;
      const provider: ExternalProviderConfig = {
        id: created.id,
        providerType: uiProviderType,
        // Now, not at the next sync, so reopening it this session knows the stored type.
        backendProviderType: created.provider_type,
        name: created.display_name,
        baseUrl: created.base_url ?? "",
        models: modelsToSave,
        availableModels: manualOnly
          ? []
          : pruneProviderModelIds(providerType, availableModels),
        maxOutputTokens: created.max_output_tokens ?? undefined,
        capabilities: created.capabilities,
        managedConfig: created.managed_config,

        hasApiKey: created.has_api_key,
        hasKaggleApiToken: created.has_kaggle_api_token,

        authKind: created.auth_kind,
        authStatus: created.auth_status,
        isReasoningModel: supportsProviderReasoningToggle(uiProviderType)
          ? isReasoningModel
          : undefined,
        createdAt,
        updatedAt,
      };
      onProvidersChange([
        ...providers.filter((p) => p.id !== created.id),
        provider,
      ]);
      if (isKaggleTPU) {
        // Persist and display the connection before starting a network action.
        // A failed start can then be retried without creating duplicate rows.
        await editProvider(provider);
        setKaggleActionPending(true);
        try {
          const status = await startKaggleTPU(created.id);
          setKaggleStatus(status);
          if (status.state === "ERROR") toast.error(status.message);
          else toast.success("Connection saved. Kaggle TPU is starting; progress is shown below.");
        } catch (error) {
          toast.error(`Connection saved. ${error instanceof Error ? error.message : "Start failed; retry Start."}`);
        } finally {
          setKaggleActionPending(false);
        }
      } else {
        resetForm();
        autoOpenedAddFormRef.current = true;
        setPage("list");
        toast.success("Connection added.");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to add connection: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function saveProviderEdits() {
    if (!editingProviderId) return;
    const existing = providers.find(
      (provider) => provider.id === editingProviderId,
    );
    if (!existing) {
      toast.error("Connection not found.");
      return;
    }
    const isEditingCustomProvider =
      isCustomProviderType(existing.providerType);
    const credentialEdit = resolveProviderCredentialEdit(
      Boolean(
        existing.hasApiKey ||
          (!existing.hasApiKey && getExternalProviderApiKey(existing.id).trim()),
      ),
      apiKey,
      clearApiKeyRequested,
    );
    const kaggleCredentialEdit = resolveProviderCredentialEdit(
      Boolean(existing.hasKaggleApiToken),
      kaggleApiToken,
      clearKaggleApiTokenRequested,
    );
    const editingContract = registryByType.get(
      toExternalBackendProviderType(existing.providerType),
    );
    const isEditingOAuthProvider = existing.authKind === "chatgpt_oauth";
    if (
      !isEditingCustomProvider &&
      !isEditingOAuthProvider &&
      existing.providerType !== "kaggle_tpu" &&
      credentialEdit.action === "missing"
    ) {
      toast.error("API key is required.");
      return;
    }
    if (
      existing.providerType === "kaggle_tpu" &&
      kaggleCredentialEdit.action === "missing"
    ) {
      toast.error("Kaggle API token is required.");
      return;
    }
    const entry = registryByType.get(existing.providerType);
    const curated = entry?.model_list_mode === "curated";
    const manualOnly =
      (isEditingCustomProvider &&
        !supportsRemoteModelCatalog(existing.providerType)) ||
      (curated && editingContract?.model_ids_editable !== false);
    const remoteAllowsManual = allowsManualModelIdsWithCatalog(
      existing.providerType,
    );
    const manualIds = parseManualModelIds(manualModelIds);
    const allowManual = manualOnly || remoteAllowsManual;
    const modelsToSave = pruneProviderModelIds(
      existing.providerType,
      allowManual
      ? [
          ...new Set([
            ...selectedModelIds,
            ...manualIds,
          ]),
        ]
      : [...selectedModelIds],
    );
    if (manualOnly) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else if (remoteAllowsManual) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else {
      if (availableModels.length === 0) {
        toast.error(
          "Load available models first, then choose which to enable.",
        );
        return;
      }
      if (selectedModelIds.length === 0) {
        toast.error("Select at least one model.");
        return;
      }
    }
    setMutatingProvider(true);
    try {
      const baseUrl = parseBaseUrlForProvider(
        baseUrlDraft,
        isEditingCustomProvider && existing.providerType !== "kaggle_tpu",
        existing.providerType,
      );
      const maxOutputTokens = supportsMaxOutputTokens
        ? parseMaxOutputTokens(maxOutputTokensDraft)
        : undefined;
      const updated = await updateProviderConfig(editingProviderId, {
        displayName: isEditingCustomProvider
          ? customProviderName.trim() ||
            customProviderDisplayName(existing.providerType)
          : existing.name,
        baseUrl,
        models: modelsToSave,
        availableModels: manualOnly
          ? []
          : pruneProviderModelIds(existing.providerType, availableModels),
        maxOutputTokens,
        capabilities: buildConnectionCapabilities(),
        managedConfig: buildKaggleManagedConfig(),
        ...(credentialEdit.action === "replace"
          ? { apiKey: credentialEdit.apiKey }
          : credentialEdit.action === "clear"
            ? { clearApiKey: true }
            : {}),
        ...(kaggleCredentialEdit.action === "replace"
          ? { kaggleApiToken: kaggleCredentialEdit.apiKey }
          : kaggleCredentialEdit.action === "clear"
            ? { clearKaggleApiToken: true }
            : {}),
      });

      if (
        credentialEdit.action === "replace" ||
        credentialEdit.action === "clear"
      ) {
        removeExternalProviderApiKey(editingProviderId);
      }
      const updatedAt = Number.isFinite(Date.parse(updated.updated_at))
        ? Date.parse(updated.updated_at)
        : Date.now();
      onProvidersChange(
        providers.map((provider) =>
          provider.id === editingProviderId
            ? {
                ...provider,
                backendProviderType: updated.provider_type,
                name: updated.display_name,
                baseUrl: updated.base_url ?? "",
                models: modelsToSave,
                availableModels: manualOnly
                  ? []
                  : pruneProviderModelIds(existing.providerType, availableModels),
                maxOutputTokens: updated.max_output_tokens ?? undefined,
                capabilities: updated.capabilities,
                managedConfig: updated.managed_config,

                hasApiKey: updated.has_api_key,
                hasKaggleApiToken: updated.has_kaggle_api_token,
                isReasoningModel: supportsProviderReasoningToggle(
                  existing.providerType,
                )
                  ? isReasoningModel
                  : undefined,
                updatedAt,
              }
            : provider,
        ),
      );
      toast.success("Connection updated.");
      resetForm();
      autoOpenedAddFormRef.current = true;
      setPage("list");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to update connection: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function editProvider(provider: ExternalProviderConfig) {
    setEditingProviderId(provider.id);
    autoOpenedAddFormRef.current = true;
    setPage("form");
    setProviderType(provider.providerType);
    setCustomProviderName(
      provider.name || customProviderDisplayName(provider.providerType),
    );
    setApiKey(
      provider.hasApiKey ? "" : getExternalProviderApiKey(provider.id),
    );

    setClearApiKeyRequested(false);
    setShowApiKey(false);
    setBaseUrlDraft(provider.baseUrl);
    // Seeded at the floor: parseMaxOutputTokens throws below it, so a row stored under
    // one would fail every unrelated edit. The resolver already reads it as the floor.
    setMaxOutputTokensDraft(
      provider.maxOutputTokens == null
        ? ""
        : Math.max(
            provider.maxOutputTokens,
            getExternalMinOutputTokens(provider.providerType),
          ).toString(),
    );
    setEditingBackendProviderType(provider.backendProviderType ?? null);
    setModelSearchQuery("");
    setIsReasoningModel(
      supportsProviderReasoningToggle(provider.providerType)
        ? (provider.capabilities?.supports_reasoning ??
            provider.isReasoningModel === true)
        : false,
    );
    setSupportsToolCalling(
      provider.capabilities?.supports_tool_calling ?? true,
    );
    setSupportsVision(provider.capabilities?.supports_vision ?? false);
    setContextLengthDraft(
      typeof provider.capabilities?.context_length === "number"
        ? provider.capabilities.context_length.toString()
        : "",
    );
    setKaggleApiToken("");
    setShowKaggleApiToken(false);
    setClearKaggleApiTokenRequested(false);
    setKaggleAutoStart(provider.managedConfig?.auto_start ?? false);
    setKaggleAutoStop(provider.managedConfig?.auto_stop ?? false);
    setKaggleTextOnly(provider.managedConfig?.text_only ?? false);
    setKaggleFastStart(provider.managedConfig?.fast_start ?? false);
    setKaggleKeepaliveDraft(
      (provider.managedConfig?.keepalive_minutes ?? 480).toString(),
    );
    if (
      isCustomProviderType(provider.providerType) &&
      !supportsRemoteModelCatalog(provider.providerType)
    ) {
      setAvailableModels([]);
      setSelectedModelIds([]);
      setManualModelIds(provider.models.join("\n"));
      return;
    }
    if (supportsRemoteModelCatalog(provider.providerType)) {
      const cachedCatalog = provider.availableModels ?? [];
      const catalogModels = pruneProviderModelIds(provider.providerType, [
        ...new Set(
          cachedCatalog
            .map((model) => model.trim())
            .filter((model) => model.length > 0),
        ),
      ]);
      setAvailableModels(catalogModels);
      const catalogSet = new Set(catalogModels);
      setSelectedModelIds(
        provider.models.filter((model) => catalogSet.has(model)),
      );
      setManualModelIds(
        provider.models.filter((model) => !catalogSet.has(model)).join("\n"),
      );
      return;
    }
    const entry = registryByType.get(provider.providerType);
    if (entry?.model_list_mode === "curated") {
      const defaults = new Set(entry.default_models);
      const inDefaults = provider.models.filter((m) => defaults.has(m));
      const custom = provider.models.filter((m) => !defaults.has(m));
      setAvailableModels([...entry.default_models]);
      setSelectedModelIds(inDefaults);
      setManualModelIds(custom.join("\n"));
    } else {
      const shortlist = entry?.default_models ?? [];
      const cachedCatalog = provider.availableModels ?? [];
      const mergedModels = pruneProviderModelIds(provider.providerType, [
        ...new Set(
          [...shortlist, ...cachedCatalog, ...provider.models]
            .map((model) => model.trim())
            .filter((model) => model.length > 0),
        ),
      ]);
      setAvailableModels(mergedModels);
      setSelectedModelIds(
        provider.models.filter((model) => mergedModels.includes(model)),
      );
      setManualModelIds("");
    }
  }

  async function deleteProvider(providerId: string) {
    setMutatingProvider(true);
    try {
      await deleteProviderConfig(providerId);
      removeExternalProviderApiKey(providerId);
      onProvidersChange(
        providers.filter((provider) => provider.id !== providerId),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to delete connection: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function testProvider(provider: ExternalProviderConfig) {

    if (provider.authKind === "chatgpt_oauth") {
      if (provider.authStatus === "connected") {
        toast.success("ChatGPT subscription is connected.");
      } else {
        await editProvider(provider);
        toast.info("Authorize this ChatGPT subscription connection.");
      }
      return;
    }
    const savedKey = provider.hasApiKey
      ? ""
      : getExternalProviderApiKey(provider.id).trim();
    // Hosted registry providers require keys. Local OpenAI-compatible presets
    // may be keyless.
    if (
      !savedKey &&
      !provider.hasApiKey &&
      !supportsRemoteModelCatalog(provider.providerType)
    ) {
      if (isCustomProviderType(provider.providerType)) {
        await editProvider(provider);
        toast.info(CUSTOM_PROVIDER_MISSING_KEY_MESSAGE);
        return;
      }
      await editProvider(provider);
      toast.info(`No API key for ${provider.name}. Add one in Connections and save.`);
      return;
    }
    try {
      const result = await testProviderConnection({
        providerType:
          toExternalBackendProviderType(provider.providerType) ??
          provider.providerType,

        providerId: provider.id,
        apiKey: savedKey,
        baseUrl: provider.baseUrl || null,
        modelId:
          provider.providerType === LEGACY_CUSTOM_PROVIDER_TYPE
            ? (provider.models[0] ?? null)
            : null,
      });
      if (result.success) {
        toast.success(result.message);
      } else {
        if (
          isCustomProviderType(provider.providerType) &&
          result.message.includes("Illegal header value b'Bearer '")
        ) {
          toast.error(CUSTOM_PROVIDER_MISSING_KEY_MESSAGE);
          return;
        }
        toast.error(result.message);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      if (
        isCustomProviderType(provider.providerType) &&
        message.includes("Illegal header value b'Bearer '")
      ) {
        toast.error(CUSTOM_PROVIDER_MISSING_KEY_MESSAGE);
        return;
      }
      toast.error(`Test failed: ${message}`);
    }
  }

  if (page === "form") {
    return (
      <div className="@container -mt-3 flex min-h-0 flex-col gap-2">
        <header className="flex items-center gap-2 pr-8">
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="size-8 rounded-[8px]"
            onClick={closeForm}
            aria-label="Back to connections"
            title="Back to connections"
          >
            <HugeiconsIcon icon={ArrowLeft02Icon} className="size-4" />
          </Button>
          <div className="flex min-w-0 items-center gap-2 leading-none">
            <span className="text-xs font-medium text-muted-foreground">
              Connections
            </span>
            <span className="size-1 rounded-full bg-muted-foreground/35" />
            <span className="truncate text-xs font-medium text-muted-foreground">
              {editingProviderId ? "Edit" : "New"}
            </span>
          </div>
        </header>

        <div className="flex max-w-[760px] flex-col gap-3">
          <section className="overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12]">
            <div className="divide-y divide-border/60">
              <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                <div className="flex min-w-0 flex-col gap-0.5">
                  <Label
                    htmlFor="provider-preset"
                    className="text-sm font-medium"
                  >
                    Connection
                  </Label>
                  <p className="text-xs leading-snug text-muted-foreground">
                    OpenAI, Anthropic, or a compatible local endpoint.
                  </p>
                </div>
                <Select
                  value={providerType}
                  onValueChange={(value) => {
                    if (editingProviderId) return;
                    setProviderType(value);
                    setAvailableModels([]);
                    setSelectedModelIds([]);
                    setManualModelIds("");
                    setModelSearchQuery("");
                    if (isCustomProviderType(value)) {
                      setCustomProviderName(customProviderDisplayName(value));
                    }
                  }}
                >
                  <SelectTrigger
                    id="provider-preset"
                    className="h-9 w-full text-sm"
                    disabled={editingProviderId != null}
                  >
                    <SelectValue placeholder="Choose a connection" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {CUSTOM_PROVIDER_PRESETS.map((preset) => (
                        <SelectItem
                          key={preset.providerType}
                          value={preset.providerType}
                        >
                          <span className="flex items-center gap-2">
                            <ApiProviderLogo
                              providerType={preset.providerType}
                              className="size-4"
                              title={preset.displayName}
                            />
                            {preset.displayName}
                          </span>
                        </SelectItem>
                      ))}
                      <SelectItem value={LEGACY_CUSTOM_PROVIDER_TYPE}>
                        <span className="flex items-center gap-2">
                          <ApiProviderLogo
                            providerType={LEGACY_CUSTOM_PROVIDER_TYPE}
                            className="size-4"
                            title={CUSTOM_PROVIDER_DISPLAY_NAME}
                          />
                          {CUSTOM_PROVIDER_DISPLAY_NAME}
                        </span>
                      </SelectItem>
                    </SelectGroup>
                    <SelectSeparator />
                    <SelectGroup>
                      {registry
                        .filter(
                          (entry) =>
                            !entry.hidden &&
                            !HIDDEN_PROVIDER_TYPES.has(entry.provider_type),
                        )
                        .map((entry) => (
                          <SelectItem
                            key={entry.provider_type}
                            value={entry.provider_type}
                          >
                            <span className="flex items-center gap-2">
                              <ApiProviderLogo
                                providerType={entry.provider_type}
                                className="size-4"
                                title={entry.display_name}
                              />
                              {entry.display_name}
                            </span>
                          </SelectItem>
                        ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>

              {showApiKeyField ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label
                      htmlFor="provider-api-key"
                      className="text-sm font-medium"
                    >
                      API key {isCustomProvider ? "(optional)" : ""}
                    </Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      {editingProviderHasSavedKey
                        ? "Saved securely. Leave blank to keep it."
                        : "Saved securely after you connect."}
                    </p>
                  </div>
                  <div className="relative min-w-0">
                    <Input
                      id="provider-api-key"
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(event) => {
                        setApiKey(event.target.value);
                        if (event.target.value.trim()) setClearApiKeyRequested(false);
                      }}
                      placeholder={
                        editingProviderHasSavedKey
                          ? "Leave blank to keep saved key"
                          : "Enter API key"
                      }
                      className="h-9 pr-9 text-sm"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKey((visible) => !visible)}
                      className="absolute top-1/2 right-1.5 flex size-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      aria-label={showApiKey ? "Hide API key" : "Show API key"}
                      aria-pressed={showApiKey}
                    >
                      {showApiKey ? (
                        <Eye className="size-3.5" />
                      ) : (
                        <EyeOff className="size-3.5" />
                      )}
                    </button>
                    {editingProviderHasSavedKey ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="mt-1 h-7 px-2 text-xs"
                        onClick={() => {
                          setApiKey("");
                          setClearApiKeyRequested((requested) => !requested);
                        }}
                      >
                        {clearApiKeyRequested ? "Keep saved key" : "Remove saved key"}
                      </Button>
                    ) : null}
                    {clearApiKeyRequested ? (
                      <p className="mt-1 text-xs text-destructive">
                        The saved key will be removed when you save this connection.
                      </p>
                    ) : null}

                  </div>
                </div>
              ) : null}

              {isCustomProvider ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <Label
                    htmlFor="provider-custom-name"
                    className="text-sm font-medium"
                  >
                    Connection name
                  </Label>
                  <Input
                    id="provider-custom-name"
                    type="text"
                    value={customProviderName}
                    onChange={(event) =>
                      setCustomProviderName(event.target.value)
                    }
                    placeholder={CUSTOM_PROVIDER_DISPLAY_NAME}
                    className="h-9 text-sm"
                  />
                </div>
              ) : null}

              {isCustomProvider &&
              selectedProviderContract?.base_url_editable !== false ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label
                      htmlFor="provider-base-url"
                      className="text-sm font-medium"
                    >
                      Base URL
                    </Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      OpenAI-compatible endpoint.
                    </p>
                  </div>
                  <Input
                    id="provider-base-url"
                    type="text"
                    value={baseUrlDraft}
                    onChange={(event) => setBaseUrlDraft(event.target.value)}
                    placeholder={customProviderBaseUrlPlaceholder(providerType)}
                    className="h-9 text-sm"
                  />
                </div>
              ) : null}

              {showsConnectionCapabilities ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-start gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label htmlFor="provider-context-length">
                      Capabilities
                    </Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      Used by the composer, tool loop and rolling context
                      manager.
                    </p>
                  </div>
                  <div className="space-y-3">
                    <Input
                      id="provider-context-length"
                      type="text"
                      inputMode="numeric"
                      value={contextLengthDraft}
                      onChange={(event) =>
                        setContextLengthDraft(event.target.value)
                      }
                      placeholder={isKaggleTPU ? "262144" : "Context tokens"}
                      className="h-9 text-sm"
                    />
                    <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
                      <label
                        htmlFor="provider-supports-tools"
                        className="flex items-center gap-2"
                      >
                        <Checkbox
                          id="provider-supports-tools"
                          checked={supportsToolCalling}
                          onCheckedChange={(checked) =>
                            setSupportsToolCalling(checked === true)
                          }
                        />
                        Studio tools
                      </label>
                      <label
                        htmlFor="provider-supports-reasoning"
                        className="flex items-center gap-2"
                      >
                        <Checkbox
                          id="provider-supports-reasoning"
                          checked={isKaggleTPU || isReasoningModel}
                          disabled={isKaggleTPU}
                          onCheckedChange={(checked) =>
                            setIsReasoningModel(checked === true)
                          }
                        />
                        Reasoning
                      </label>
                      <label
                        htmlFor="provider-supports-vision"
                        className="flex items-center gap-2"
                      >
                        <Checkbox
                          id="provider-supports-vision"
                          checked={!kaggleTextOnly && supportsVision}
                          disabled={isKaggleTPU && kaggleTextOnly}
                          onCheckedChange={(checked) =>
                            setSupportsVision(checked === true)
                          }
                        />
                        Vision
                      </label>
                    </div>
                  </div>
                </div>
              ) : null}

              {isKaggleTPU ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-start gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label htmlFor="kaggle-api-token">Kaggle API token</Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      The launcher is included with Studio. Generate a token in
                      Kaggle Settings → API; it is stored securely on this Mac.
                    </p>
                  </div>
                  <div className="space-y-3">
                    <div className="relative min-w-0">
                      <Input
                        id="kaggle-api-token"
                        type={showKaggleApiToken ? "text" : "password"}
                        value={kaggleApiToken}
                        onChange={(event) => {
                          setKaggleApiToken(event.target.value);
                          if (event.target.value.trim()) {
                            setClearKaggleApiTokenRequested(false);
                          }
                        }}
                        placeholder={
                          editingProviderHasSavedKaggleToken
                            ? "Leave blank to keep saved token"
                            : "Paste Kaggle API token"
                        }
                        className="h-9 pr-9 text-sm"
                      />
                      <button
                        type="button"
                        onClick={() =>
                          setShowKaggleApiToken((visible) => !visible)
                        }
                        className="absolute top-1/2 right-1.5 flex size-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        aria-label={
                          showKaggleApiToken
                            ? "Hide Kaggle API token"
                            : "Show Kaggle API token"
                        }
                        aria-pressed={showKaggleApiToken}
                      >
                        {showKaggleApiToken ? (
                          <Eye className="size-3.5" />
                        ) : (
                          <EyeOff className="size-3.5" />
                        )}
                      </button>
                    </div>
                    {editingProviderHasSavedKaggleToken ? (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => {
                          setKaggleApiToken("");
                          setClearKaggleApiTokenRequested(
                            (requested) => !requested,
                          );
                        }}
                      >
                        {clearKaggleApiTokenRequested
                          ? "Keep saved token"
                          : "Remove saved token"}
                      </Button>
                    ) : null}
                    {clearKaggleApiTokenRequested ? (
                      <p className="text-xs text-destructive">
                        The saved Kaggle token will be removed when you save.
                      </p>
                    ) : null}
                    <div className="grid grid-cols-2 gap-2 @max-[520px]:grid-cols-1">
                      <label
                        htmlFor="kaggle-auto-start"
                        className="flex items-center gap-2 text-sm"
                      >
                        <Checkbox
                          id="kaggle-auto-start"
                          checked={kaggleAutoStart}
                          onCheckedChange={(checked) =>
                            setKaggleAutoStart(checked === true)
                          }
                        />
                        Auto-start when Studio opens
                      </label>
                      <label
                        htmlFor="kaggle-auto-stop"
                        className="flex items-center gap-2 text-sm"
                      >
                        <Checkbox
                          id="kaggle-auto-stop"
                          checked={kaggleAutoStop}
                          onCheckedChange={(checked) =>
                            setKaggleAutoStop(checked === true)
                          }
                        />
                        Auto-stop with Studio
                      </label>
                      <label
                        htmlFor="kaggle-text-only"
                        className="flex items-center gap-2 text-sm"
                      >
                        <Checkbox
                          id="kaggle-text-only"
                          checked={kaggleTextOnly}
                          onCheckedChange={(checked) =>
                            setKaggleTextOnly(checked === true)
                          }
                        />
                        Text only
                      </label>
                      <label
                        htmlFor="kaggle-fast-start"
                        className="flex items-center gap-2 text-sm"
                      >
                        <Checkbox
                          id="kaggle-fast-start"
                          checked={kaggleFastStart}
                          onCheckedChange={(checked) =>
                            setKaggleFastStart(checked === true)
                          }
                        />
                        Fast start
                      </label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Label
                        htmlFor="kaggle-keepalive"
                        className="shrink-0 text-xs"
                      >
                        Keepalive (minutes)
                      </Label>
                      <Input
                        id="kaggle-keepalive"
                        type="text"
                        inputMode="numeric"
                        value={kaggleKeepaliveDraft}
                        onChange={(event) =>
                          setKaggleKeepaliveDraft(event.target.value)
                        }
                        className="h-8 max-w-28 text-sm"
                      />
                    </div>
                    {editingProviderId ? (
                      <div className="rounded-[8px] border border-border/70 bg-background/45 p-3">
                        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <p className="text-xs font-medium">
                              {kaggleStatus?.state ?? "DISCONNECTED"}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {kaggleStatus?.message ??
                                "Status not available yet."}
                            </p>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <Button
                              type="button"
                              size="sm"
                              className="h-7"
                              disabled={kaggleActionPending || ["STARTING", "PROVISIONING", "LOADING", "READY", "STOPPING"].includes(kaggleStatus?.state ?? "")}
                              onClick={() => void runKaggleAction("start")}
                            >
                              Start
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7"
                              disabled={kaggleActionPending || ["STARTING", "PROVISIONING", "LOADING", "STOPPING"].includes(kaggleStatus?.state ?? "")}
                              onClick={() => void runKaggleAction("reconnect")}
                            >
                              Reconnect
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7"
                              disabled={kaggleActionPending}
                              onClick={() => void runKaggleAction("stop")}
                            >
                              Stop
                            </Button>
                          </div>
                        </div>
                        {kaggleStatus?.base_url ? (
                          <p className="break-all font-mono text-[11px] text-muted-foreground">
                            {kaggleStatus.base_url}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {supportsMaxOutputTokens ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-start gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label
                      htmlFor="provider-max-output-tokens"
                      className="text-sm font-medium"
                    >
                      Max Tokens limit
                    </Label>
                    <p
                      id="provider-max-output-tokens-help"
                      className="text-xs leading-snug text-muted-foreground"
                    >
                      Caps Max Tokens for this connection. Never raises it past a
                      model's documented limit. Leave blank to use that limit, or
                      32,768 for a model without one.
                    </p>
                  </div>
                  <div className="flex min-w-0 flex-col gap-1.5">
                    {/*
                      A TEXT input, deliberately, matching NumericValueInput in the
                      run-settings panel. `type="number"` runs the HTML value
                      sanitization algorithm, which replaces anything the engine does
                      not read as a valid floating-point number with the EMPTY STRING
                      (WHATWG HTML 4.10.5). Blank means "no override" here, so a
                      grouped or localised entry such as "131,072" would leave the box
                      looking filled, report "" to React, and silently CLEAR the
                      user's override on save with no error. Keeping the raw string
                      lets `parseMaxOutputTokens` reject it and say why.
                    */}
                    <Input
                      id="provider-max-output-tokens"
                      type="text"
                      inputMode="numeric"
                      autoComplete="off"
                      spellCheck={false}
                      value={maxOutputTokensDraft}
                      onChange={(event) =>
                        setMaxOutputTokensDraft(event.target.value)
                      }
                      placeholder="32768"
                      aria-describedby="provider-max-output-tokens-help provider-max-output-tokens-warning"
                      className="h-9 text-sm"
                    />
                    <p
                      id="provider-max-output-tokens-warning"
                      className="text-xs leading-snug text-amber-700 dark:text-amber-400"
                    >
                      If the upstream provider does not support this value,
                      requests may fail.
                    </p>
                  </div>
                </div>
              ) : null}

              {showReasoningToggle ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <Label
                    htmlFor="provider-is-reasoning"
                    className="text-sm font-medium"
                  >
                    Reasoning model
                  </Label>
                  <label
                    htmlFor="provider-is-reasoning"
                    className="flex cursor-pointer items-center gap-2 text-sm"
                  >
                    <Checkbox
                      id="provider-is-reasoning"
                      checked={isReasoningModel}
                      onCheckedChange={(checked) =>
                        setIsReasoningModel(checked === true)
                      }
                    />
                    This server runs a reasoning model
                  </label>
                </div>
              ) : null}
              {runsStudioToolsLocally ? (
                <div className="px-4 py-3">
                  <p className="text-xs text-muted-foreground">
                    Models on this connection can use Studio&apos;s Search, Code,
                    MCP and Docs tools. Those run on this machine, and their
                    results are sent back to the provider as part of the next
                    message. Code and terminal calls still ask before anything
                    risky runs.
                  </p>
                </div>
              ) : null}
            </div>
          </section>


          {isCodexSubscription ? (
            <OpenAICodexConnect
              providerId={editingProviderId}
              authStatus={providers.find((provider) => provider.id === editingProviderId)?.authStatus}
              ensureProvider={ensureCodexProvider}
              onChanged={async () => {
                const synced = await syncExternalProvidersFromBackend(providersRef.current);
                providersRef.current = synced;
                onProvidersChange(synced);
              }}
            />
          ) : null}

          <section className="overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12]">
            <AnimatePresence initial={false} mode="wait">
              <motion.div
                key={modelsPanelKey}
                className="origin-top overflow-hidden"
                initial={reduceMotion ? false : { opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={reduceMotion ? undefined : { opacity: 0, height: 0 }}
                transition={{
                  height: {
                    duration: PROVIDER_FORM_DURATION,
                    ease: PROVIDER_FORM_EASE,
                  },
                  opacity: {
                    duration: reduceMotion ? 0 : 0.14,
                    ease: PROVIDER_FORM_EASE,
                  },
                }}
              >
                <div
                  className={`flex flex-wrap items-center justify-between gap-3 px-4 py-3 ${showModelsBody ? "border-border/60 border-b" : ""}`}
                >
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label className="text-sm font-medium">Models</Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      {modelStatusLabel}
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className={
                      availableModels.length > 0
                        ? "h-7 shrink-0 border-transparent bg-transparent px-2 text-xs text-muted-foreground shadow-none hover:bg-muted/45 hover:text-foreground"
                        : "h-8 shrink-0 px-3"
                    }
                    disabled={loadModelsDisabled}
                    title={loadModelsTitle}
                    onClick={() => void loadModels()}
                  >
                    {modelsLoading ? (
                      <>
                        <Spinner className="mr-2 size-3.5" />
                        Loading…
                      </>
                    ) : availableModels.length > 0 ? (
                      "Reload models"
                    ) : (
                      "Load available models"
                    )}
                  </Button>
                </div>
                {isCustomProvider && !supportsRemoteModelCatalog(providerType) ? (
                  <div className="space-y-3 px-4 py-4">
                    <div className="space-y-2">
                      <Label
                        htmlFor="provider-manual-models"
                        className="text-sm font-medium"
                      >
                        Model IDs (one per line or comma-separated)
                      </Label>
                      <Textarea
                        id="provider-manual-models"
                        value={manualModelIds}
                        onChange={(event) =>
                          setManualModelIds(event.target.value)
                        }
                        placeholder={customProviderModelIdsPlaceholder(providerType)}
                        rows={5}
                        className="min-h-[100px] resize-y font-mono text-sm"
                      />
                    </div>
                  </div>
                ) : isCuratedModelList ? (
                  <div className="space-y-3 px-4 py-4">
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      Select from suggestions below or enter exact model IDs.
                    </p>
                    {availableModels.length > 0 ? (
                      <div className="space-y-3 rounded-[8px] border border-border/70 bg-background/50 p-3">
                        <div className="grid grid-cols-[minmax(90px,auto)_minmax(0,1fr)_auto] items-center gap-3 @max-[520px]:grid-cols-1">
                          <span className="whitespace-nowrap text-xs font-medium text-muted-foreground">
                            {availableModelsLabel}
                          </span>
                          <Input
                            id={`provider-model-search-${modelsPanelKey}`}
                            type="search"
                            value={modelSearchQuery}
                            onChange={(event) =>
                              setModelSearchQuery(event.target.value)
                            }
                            placeholder="Search"
                            aria-label="Search models"
                            className={modelSearchInputClassName}
                          />
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={selectAllModels}
                            >
                              Select all
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={() => {
                                clearModelSelection();
                                setManualModelIds("");
                              }}
                            >
                              Clear
                            </Button>
                          </div>
                        </div>
                        <ul className="max-h-56 overflow-y-auto rounded-[8px] border border-border/70 bg-background/50">
                          {filteredAvailableModels.length === 0 ? (
                            <li className="px-3 py-3 text-xs text-muted-foreground">
                              No matching models
                            </li>
                          ) : (
                            filteredAvailableModels.map((model, index) => (
                              <li
                                key={model}
                                className="flex cursor-pointer items-center gap-2.5 border-border/60 border-b px-3 py-2 last:border-b-0 hover:bg-muted/35"
                                onClick={() => toggleModel(model)}
                              >
                                <Checkbox
                                  id={`provider-model-curated-${modelsPanelKey}-${index}`}
                                  checked={selectedModelIds.includes(model)}
                                  onCheckedChange={() => toggleModel(model)}
                                  onClick={(event) => event.stopPropagation()}
                                />
                                <span
                                  className="min-w-0 break-all text-sm leading-tight"
                                >
                                  {model}
                                </span>
                              </li>
                            ))
                          )}
                        </ul>
                      </div>
                    ) : null}
                    {modelIdsEditable ? (
                      <div className="space-y-2">
                        <Label
                          htmlFor="provider-manual-models"
                          className="text-sm font-medium"
                        >
                          Model IDs (one per line or comma-separated)
                        </Label>
                        <Textarea
                          id="provider-manual-models"
                          value={manualModelIds}
                          onChange={(event) => setManualModelIds(event.target.value)}
                          placeholder={"model-id-1\nmodel-id-2"}
                          rows={5}
                          className="min-h-[100px] resize-y font-mono text-sm"
                        />
                      </div>
                    ) : null}
                  </div>
                ) : availableModels.length === 0 &&
                  !allowsManualModelIdsWithCatalog(providerType) ? null : (
                  <div className="space-y-3 px-4 py-4">
                    {availableModels.length === 0 ? null : (
                      <>
                        <div className="grid grid-cols-[minmax(90px,auto)_minmax(0,1fr)_auto] items-center gap-3 @max-[520px]:grid-cols-1">
                          <span className="whitespace-nowrap text-xs font-medium text-muted-foreground">
                            {availableModelsLabel}
                          </span>
                          <Input
                            id={`provider-model-search-${modelsPanelKey}`}
                            type="search"
                            value={modelSearchQuery}
                            onChange={(event) =>
                              setModelSearchQuery(event.target.value)
                            }
                            placeholder="Search"
                            aria-label="Search models"
                            className={modelSearchInputClassName}
                          />
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={selectAllModels}
                            >
                              Select all
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={clearModelSelection}
                            >
                              Clear
                            </Button>
                          </div>
                        </div>
                        <ul className="max-h-56 overflow-y-auto rounded-[8px] border border-border/70 bg-background/50">
                          {filteredAvailableModels.length === 0 ? (
                            <li className="px-3 py-3 text-xs text-muted-foreground">
                              No matching models
                            </li>
                          ) : (
                            filteredAvailableModels.map((model, index) => (
                              <li
                                key={model}
                                className="flex cursor-pointer items-center gap-2.5 border-border/60 border-b px-3 py-2 last:border-b-0 hover:bg-muted/35"
                                onClick={() => toggleModel(model)}
                              >
                                <Checkbox
                                  id={`provider-model-remote-${modelsPanelKey}-${index}`}
                                  checked={selectedModelIds.includes(model)}
                                  onCheckedChange={() => toggleModel(model)}
                                  onClick={(event) => event.stopPropagation()}
                                />
                                <span
                                  className="min-w-0 break-all text-sm leading-tight"
                                >
                                  {model}
                                </span>
                              </li>
                            ))
                          )}
                        </ul>
                      </>
                    )}
                    {/* Manual IDs allowed alongside catalog load. */}
                    {allowsManualModelIdsWithCatalog(providerType) ? (
                      <div className="space-y-2">
                        <Label
                          htmlFor="provider-manual-models"
                          className="text-sm font-medium"
                        >
                          {availableModels.length === 0
                            ? "Or enter model IDs manually (one per line or comma-separated)"
                            : "Additional model IDs (one per line or comma-separated)"}
                        </Label>
                        <Textarea
                          id="provider-manual-models"
                          value={manualModelIds}
                          onChange={(event) =>
                            setManualModelIds(event.target.value)
                          }
                          placeholder={customProviderModelIdsPlaceholder(
                            providerType,
                          )}
                          rows={4}
                          className="min-h-[80px] resize-y font-mono text-sm"
                        />
                      </div>
                    ) : null}
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </section>

          <div className="mb-3 flex flex-wrap items-center justify-end gap-3">
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                size="sm"
                className="h-8"
                disabled={
                  registryLoading ||
                  syncingProviders ||
                  modelsLoading ||
                  mutatingProvider
                }
                onClick={() =>
                  editingProviderId
                    ? void saveProviderEdits()
                    : void addProvider()
                }
              >
                {editingProviderId ? "Save connection" : "Add connection"}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                onClick={editingProviderId ? closeForm : resetForm}
              >
                {editingProviderId ? "Cancel" : "Clear"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-col gap-6">
      <header className="flex flex-col gap-1 pr-8">
        <div className="flex min-w-0 flex-col gap-1">
          <h1 className="font-heading text-lg font-semibold">Connections</h1>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Manage model connections for chat.
          </p>
        </div>
      </header>

      <div className="flex w-full max-w-[760px] flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-x-6">
        <div className="flex items-center gap-2">
          <Label
            htmlFor="chat-connections-enabled"
            className="cursor-pointer text-xs text-muted-foreground"
          >
            Enable connections
          </Label>
          <Switch
            id="chat-connections-enabled"
            checked={connectionsEnabled}
            onCheckedChange={setConnectionsEnabled}
            aria-label="Enable connections"
            aria-describedby="chat-connections-description"
          />
        </div>
        <p
          id="chat-connections-description"
          className="max-w-md text-ui-11 leading-snug text-muted-foreground/65 sm:text-right"
        >
          When off, all connections are disabled.
        </p>
      </div>

      <section className="flex max-w-[760px] flex-col gap-2">
        <div className="overflow-hidden rounded-[14px] border border-border/70 bg-muted/[0.12]">
          <button
            type="button"
            onClick={openAddProvider}
            className="group/add flex w-full items-center justify-between gap-3 border-border/60 border-b px-3 py-2.5 text-left text-sm font-medium text-muted-foreground transition-colors hover:bg-muted/35 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
          >
            <span className="flex min-w-0 items-center gap-2 rounded-full border border-border bg-background/50 px-3 py-1.5 transition-colors group-hover/add:border-control-accent/25 group-hover/add:text-control-accent">
              <HugeiconsIcon icon={PlusSignIcon} className="size-4 shrink-0" />
              <span>Add connection</span>
            </span>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground/90">
              {providers.length} connections · {totalModels} models
            </span>
          </button>
          {providers.length === 0 ? (
            <div className="px-3 py-4">
              <div className="flex min-w-0 flex-col gap-0.5">
                <span className="text-sm font-medium text-foreground">
                  No connections yet
                </span>
                <span className="text-xs leading-snug text-muted-foreground">
                  Add a connection to use hosted models from chat.
                </span>
              </div>
            </div>
          ) : (
            <>
              {providers.map((provider) => {
                const registryEntry = registryByType.get(provider.providerType);
                const detail =
                  provider.baseUrl || registryEntry?.base_url || "";
                const providerLabel =
                  registryEntry?.display_name ??
                  customProviderDisplayName(provider.providerType);
                const modelSummary = formatModelSummary(provider.models);
                return (
                  <div
                    key={provider.id}
                    className="group grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-border/60 border-b px-3 py-3 transition-colors last:border-b-0 hover:bg-muted/35 max-sm:grid-cols-1"
                  >
                    <div className="flex min-w-0 items-start gap-3">
                      <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-[8px] border border-border/70 bg-background/80">
                        <ApiProviderLogo
                          providerType={provider.providerType}
                          className="size-5"
                          title={provider.name}
                        />
                      </div>
                      <div className="min-w-0 pt-px">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate text-sm font-medium text-foreground">
                            {provider.name}
                          </span>
                          <span className="shrink-0 rounded-[6px] border border-control-accent/15 bg-control-accent/8 px-1.5 py-0.5 text-ui-10 leading-none text-control-accent">
                            {provider.models.length}{" "}
                            {provider.models.length === 1 ? "model" : "models"}
                          </span>
                        </div>
                        <div className="mt-0.5 truncate text-xs text-muted-foreground">
                          <span>{providerLabel}</span>
                          {detail ? (
                            <>
                              {" · "}
                              <span>{detail}</span>
                            </>
                          ) : null}
                        </div>
                        <div
                          className="mt-1 truncate text-ui-11 leading-4 text-muted-foreground/80"
                          title={provider.models.join(", ")}
                        >
                          {modelSummary}
                        </div>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center justify-end gap-0.5 text-muted-foreground">
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        className="size-7 rounded-[8px] hover:text-foreground"
                        disabled={mutatingProvider}
                        onClick={() => editProvider(provider)}
                        title="Edit connection"
                        aria-label={`Edit ${provider.name}`}
                      >
                        <HugeiconsIcon icon={Edit03Icon} className="size-4" />
                      </Button>
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        className="size-7 rounded-[8px] hover:text-foreground"
                        disabled={mutatingProvider}
                        onClick={() => void testProvider(provider)}
                        title="Check connection"
                        aria-label={`Check ${provider.name}`}
                      >
                        <HugeiconsIcon icon={Wifi02Icon} className="size-4" />
                      </Button>
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        className="size-7 rounded-[8px] hover:text-destructive"
                        disabled={mutatingProvider}
                        onClick={() => void deleteProvider(provider.id)}
                        title="Delete connection"
                        aria-label={`Delete ${provider.name}`}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      </section>
    </div>
  );
}

interface ChatProvidersDialogProps extends ChatProvidersSettingsProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ChatProvidersDialog({
  open,
  onOpenChange,
  providers,
  onProvidersChange,
}: ChatProvidersDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        overlayClassName="bg-black/50 backdrop-blur-sm"
        className="flex max-h-[90dvh] w-[96vw] flex-col gap-0 overflow-y-auto p-8 sm:max-w-none md:max-w-[44rem]"
      >
        <DialogHeader className="sr-only">
          <DialogTitle>Connections</DialogTitle>
          <DialogDescription>
            Manage model connections for chat.
          </DialogDescription>
        </DialogHeader>
        <ChatProvidersSettings
          providers={providers}
          onProvidersChange={onProvidersChange}
        />
      </DialogContent>
    </Dialog>
  );
}
