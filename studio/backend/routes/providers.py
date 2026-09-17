# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
API routes for external LLM provider management.

Endpoints:
  - Discover available provider types (registry)
  - CRUD for saved provider configurations and API keys
  - Fetch the RSA public key for API key encryption
  - Test provider connectivity
  - List models from a provider
"""

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)

from routes.provider_credentials import (
    current_credential_write,
    decrypt_request_secret_or_400,
    require_ui_session,
    resolve_provider_api_key_or_400,
    serialize_provider_config,
)
from core.inference.key_exchange import (
    get_public_key_fingerprint,
    get_public_key_pem,
)
from core.inference.providers import (
    effective_provider_capabilities,
    get_base_url,
    get_provider_info,
    list_available_providers,
    validate_provider_base_url,
)
from core.inference.pricing import pricing_snapshot
from core.inference.external_provider import ExternalProviderClient
from core.inference.kaggle_tpu import kaggle_tpu_manager

from core.inference import openai_codex_auth
from models.providers import (
    ProviderCreate,
    ProviderCredentialMigration,
    KaggleTPULifecycleStatus,
    ProviderModelsRequest,
    ProviderModelInfo,
    ProviderResponse,
    ProviderRegistryEntry,
    ProviderTestRequest,
    ProviderTestResult,
    ProviderUpdate,
)
from storage import credential_secrets, providers_db
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)

router = APIRouter()


def _effective_capabilities(row: dict) -> dict:
    capabilities = effective_provider_capabilities(row["provider_type"], row.get("capabilities"))
    managed = row.get("managed_config") or {}
    if row["provider_type"] == "kaggle_tpu" and managed.get("text_only"):
        capabilities["supports_vision"] = False
        capabilities["supports_images"] = False
    return capabilities


def _provider_response(row: dict) -> ProviderResponse:
    return ProviderResponse(
        id = row["id"],
        provider_type = row["provider_type"],
        display_name = row["display_name"],
        base_url = row["base_url"],
        is_enabled = bool(row["is_enabled"]),
        has_api_key = credential_secrets.has_secret(
            credential_secrets.PROVIDER_API_KEY_KIND,
            row["id"],
        ),
        has_kaggle_api_token = (
            row["provider_type"] == "kaggle_tpu"
            and credential_secrets.has_secret(
                credential_secrets.KAGGLE_API_TOKEN_KIND,
                row["id"],
            )
        ),
        auth_kind = ("chatgpt_oauth" if row["provider_type"] == "openai_codex" else "api_key"),
        auth_status = (
            openai_codex_auth.auth_status(row["id"])
            if row["provider_type"] == "openai_codex"
            else (
                "connected"
                if credential_secrets.has_secret(
                    credential_secrets.PROVIDER_API_KEY_KIND, row["id"]
                )
                else "disconnected"
            )
        ),
        models=row.get("models") or [],
        available_models=row.get("available_models") or [],
        max_output_tokens=row.get("max_output_tokens"),
        capabilities=_effective_capabilities(row),
        managed_config=row.get("managed_config"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _validate_provider_auth_contract(
    info: dict,
    *,
    encrypted_api_key: str | None,
    base_url: str | None,
    models: list[str] | None,
    updating: bool,
    clear_api_key: bool = False,
) -> None:
    if info.get("auth_kind") != "chatgpt_oauth":
        return
    if encrypted_api_key or clear_api_key:
        raise HTTPException(status_code = 400, detail = "ChatGPT subscriptions do not use API keys.")
    if base_url is not None and (not updating or base_url != info["base_url"]):
        raise HTTPException(status_code = 400, detail = "ChatGPT subscription routing is fixed.")
    if models is not None and (not models or not set(models).issubset(set(info["default_models"]))):
        raise HTTPException(status_code = 400, detail = "Choose only curated Codex models.")


def _validate_max_output_tokens_contract(
    provider_type: str,
    field_was_set: bool,
    value: Optional[int] = None,
) -> None:
    """Reject a non-null override on a ChatGPT subscription.

    Codex routing, model list and output cap are all fixed, so an override stored there
    would never be read. Every other type takes one: the frontend uses it to lower a
    model's documented cap, or to replace the 32,768-token fallback for a model with no
    documented cap.

    An explicit null is allowed everywhere, Codex included: a blank field serialises as
    null rather than as an omission, and clearing an absent override is a no-op.
    """
    if field_was_set and value is not None and provider_type == "openai_codex":
        raise HTTPException(
            status_code = 400,
            detail = "ChatGPT subscriptions use a fixed Max Tokens limit.",
        )


def _validate_provider_metadata_contract(
    provider_type: str,
    capabilities,
    managed_config,
) -> None:
    if managed_config is not None and provider_type != "kaggle_tpu":
        raise HTTPException(
            status_code=400,
            detail="Managed launcher settings are only valid for Kaggle TPU connections.",
        )
    if provider_type == "kaggle_tpu" and capabilities is not None:
        if capabilities.api_mode != "chat_completions":
            raise HTTPException(
                status_code=400,
                detail="Kaggle TPU currently uses the chat_completions API mode.",
            )
    if provider_type == "openai_compatible" and capabilities is not None:
        if capabilities.api_mode != "chat_completions":
            raise HTTPException(
                status_code=400,
                detail="OpenAI Compatible currently uses the chat_completions API mode.",
            )


def _validate_kaggle_token_contract(
    provider_type: str,
    encrypted_kaggle_api_token: str | None,
    clear_kaggle_api_token: bool = False,
) -> None:
    if (encrypted_kaggle_api_token or clear_kaggle_api_token) and provider_type != "kaggle_tpu":
        raise HTTPException(
            status_code=400,
            detail="Kaggle API tokens are only valid for Kaggle TPU connections.",
        )


# ── Public key for API key encryption ─────────────────────────────


@router.get("/public-key")
async def get_public_key(current_subject: str = Depends(get_current_subject)):
    """Return the RSA public key PEM for client-side API key encryption.

    ``fingerprint`` is a short SHA256 of the PEM; a mismatch with what the
    frontend captured at encrypt time signals the keypair rotated mid-flight.
    """
    return {
        "public_key": get_public_key_pem(),
        "fingerprint": get_public_key_fingerprint(),
    }


# ── Provider registry (static) ───────────────────────────────────


@router.get("/registry", response_model = list[ProviderRegistryEntry])
async def list_registry(
    include_hidden: bool = False, current_subject: str = Depends(get_current_subject)
):
    """List all supported provider types with their default configurations.

    ``include_hidden=true`` also returns the backend-only entries (the
    self-hosted presets), which carry the studio-tools capability the composer
    needs. It is opt-in so that a browser still running a pre-capability bundle,
    which does not know to filter on ``hidden``, keeps seeing exactly the list
    it saw before and cannot render them as duplicate dropdown options.
    """
    return list_available_providers(include_hidden = include_hidden)


# ── Per-MTok pricing snapshot for client-side cost display ──────────


@router.get("/pricing")
async def get_pricing_snapshot(current_subject: str = Depends(get_current_subject)):
    """Static per-MTok pricing table the frontend uses to convert upstream
    usage into per-turn USD cost. See ``core/inference/pricing.py`` for sourcing."""
    return pricing_snapshot()


# ── Provider config CRUD ──────────────────────────────────────────


# FastAPI offloads sync reads; mutations stay on-loop to preserve atomic sequences.
@router.get("/", response_model = list[ProviderResponse])
def list_provider_configs(_current_subject: str = Depends(get_current_subject)):
    """List all saved provider configurations."""
    rows = providers_db.list_providers()
    return [_provider_response(row) for row in rows]


@router.post("/", response_model = ProviderResponse, status_code = 201)
async def create_provider_config(
    payload: ProviderCreate,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Create a saved provider configuration and optional encrypted API key."""

    require_ui_session(via_api_key)
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}. "
            f"Use GET /api/providers/registry to see available types.",
        )

    _validate_max_output_tokens_contract(
        payload.provider_type,
        "max_output_tokens" in payload.model_fields_set,
        payload.max_output_tokens,
    )
    _validate_provider_metadata_contract(
        payload.provider_type,
        payload.capabilities,
        payload.managed_config,
    )
    _validate_kaggle_token_contract(
        payload.provider_type,
        payload.encrypted_kaggle_api_token,
    )

    _validate_provider_auth_contract(
        info,
        encrypted_api_key = payload.encrypted_api_key,
        base_url = payload.base_url,
        models = payload.models,
        updating = False,
    )

    base_url = payload.base_url or info["base_url"]
    # An empty base URL stays allowed (custom/vLLM entries carry none until the
    # user fills one in); anything present is checked before a key is decrypted.
    if base_url:
        try:
            base_url = validate_provider_base_url(base_url)
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from None

    api_key = resolve_provider_api_key_or_400(None, payload.encrypted_api_key)
    kaggle_api_token = decrypt_request_secret_or_400(
        payload.encrypted_kaggle_api_token,
        label="Kaggle API token",
    )
    provider_id = uuid.uuid4().hex[:16]

    if api_key or kaggle_api_token:
        credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        providers_db.create_provider(
            id=provider_id,
            provider_type=payload.provider_type,
            display_name=payload.display_name,
            base_url=base_url,
            models=payload.models,
            available_models=payload.available_models,
            max_output_tokens=payload.max_output_tokens,
            capabilities=(
                payload.capabilities.model_dump() if payload.capabilities is not None else None
            ),
            managed_config=(
                payload.managed_config.model_dump() if payload.managed_config is not None else None
            ),
        )
        try:
            if api_key:
                credential_secrets.save_provider_api_key(provider_id, api_key)
            if kaggle_api_token:
                credential_secrets.save_kaggle_api_token(provider_id, kaggle_api_token)
        except Exception:
            credential_secrets.delete_provider_api_key(provider_id)
            credential_secrets.delete_kaggle_api_token(provider_id)
            providers_db.delete_provider(provider_id)
            raise

    row = providers_db.get_provider(provider_id)
    return _provider_response(row)


@router.put("/{provider_id}", response_model = ProviderResponse)
@serialize_provider_config
async def update_provider_config(
    provider_id: str,
    payload: ProviderUpdate,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Update a saved provider configuration."""

    require_ui_session(via_api_key)
    existing = providers_db.get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code = 404, detail = "Provider not found")

    existing_info = get_provider_info(existing["provider_type"]) or {}
    max_output_tokens_requested = "max_output_tokens" in payload.model_fields_set
    _validate_max_output_tokens_contract(
        existing["provider_type"],
        max_output_tokens_requested,
        payload.max_output_tokens,
    )
    _validate_provider_metadata_contract(
        existing["provider_type"],
        payload.capabilities,
        payload.managed_config,
    )
    _validate_kaggle_token_contract(
        existing["provider_type"],
        payload.encrypted_kaggle_api_token,
        payload.clear_kaggle_api_token,
    )
    _validate_provider_auth_contract(
        existing_info,
        encrypted_api_key = payload.encrypted_api_key,
        base_url = payload.base_url,
        models = payload.models,
        updating = True,
        clear_api_key = payload.clear_api_key,
    )

    if payload.clear_api_key and payload.encrypted_api_key:
        raise HTTPException(
            status_code = 400,
            detail = "Cannot replace and clear an API key in the same request",
        )
    if payload.clear_kaggle_api_token and payload.encrypted_kaggle_api_token:
        raise HTTPException(
            status_code=400,
            detail="Cannot replace and clear a Kaggle API token in the same request",
        )

    metadata_fields = {
        "display_name",
        "base_url",
        "is_enabled",
        "models",
        "available_models",
        "max_output_tokens",
        "capabilities",
        "managed_config",
    }
    metadata_requested = bool(payload.model_fields_set & metadata_fields)

    # Only a *changed* base URL is validated. The dialog re-sends the stored value
    # on every edit, so validating an unchanged legacy row would lock the user out
    # of editing its models or API key. Outbound use is still checked.
    base_url = payload.base_url
    if base_url and base_url != existing["base_url"]:
        try:
            base_url = validate_provider_base_url(base_url)
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from None

    replacement_api_key = None
    if payload.encrypted_api_key:
        credential_secrets.get_or_create_credential_encryption_key()
        replacement_api_key = resolve_provider_api_key_or_400(
            provider_id, payload.encrypted_api_key
        )
        if not replacement_api_key:
            raise HTTPException(status_code = 400, detail = "API key cannot be empty")

    replacement_kaggle_api_token = None
    if payload.encrypted_kaggle_api_token:
        credential_secrets.get_or_create_credential_encryption_key()
        replacement_kaggle_api_token = decrypt_request_secret_or_400(
            payload.encrypted_kaggle_api_token,
            label="Kaggle API token",
        )
        if not replacement_kaggle_api_token:
            raise HTTPException(status_code=400, detail="Kaggle API token cannot be empty")
    kaggle_token_requested = bool(
        payload.encrypted_kaggle_api_token or payload.clear_kaggle_api_token
    )
    existing_kaggle_api_token = (
        credential_secrets.get_kaggle_api_token(provider_id)
        if kaggle_token_requested
        else None
    )

    with current_credential_write(credential):
        if metadata_requested:
            metadata_updates = dict(
                id = provider_id,
                display_name = payload.display_name,
                base_url = base_url,
                is_enabled = payload.is_enabled,
                models = payload.models,
                available_models = payload.available_models,
            )
            if max_output_tokens_requested:
                metadata_updates["max_output_tokens"] = payload.max_output_tokens
            if "capabilities" in payload.model_fields_set:
                metadata_updates["capabilities"] = (
                    payload.capabilities.model_dump() if payload.capabilities is not None else None
                )
            if "managed_config" in payload.model_fields_set:
                metadata_updates["managed_config"] = (
                    payload.managed_config.model_dump()
                    if payload.managed_config is not None
                    else None
                )
            providers_db.update_provider(**metadata_updates)
        try:
            if replacement_api_key is not None:
                credential_secrets.save_provider_api_key(provider_id, replacement_api_key)
            elif payload.clear_api_key:
                credential_secrets.delete_provider_api_key(provider_id)
            if replacement_kaggle_api_token is not None:
                credential_secrets.save_kaggle_api_token(
                    provider_id,
                    replacement_kaggle_api_token,
                )
            elif payload.clear_kaggle_api_token:
                credential_secrets.delete_kaggle_api_token(provider_id)
        except Exception:
            if metadata_requested:
                try:
                    providers_db.update_provider(
                        id=provider_id,
                        display_name=existing["display_name"],
                        base_url=existing["base_url"],
                        is_enabled=bool(existing["is_enabled"]),
                        models=existing.get("models") or [],
                        available_models=existing.get("available_models") or [],
                        max_output_tokens=existing.get("max_output_tokens"),
                        capabilities=existing.get("capabilities"),
                        managed_config=existing.get("managed_config"),
                    )
                except Exception:
                    logger.exception(
                        "provider.update_metadata_rollback_failed", provider_id = provider_id
                    )
            if kaggle_token_requested:
                try:
                    if existing_kaggle_api_token:
                        credential_secrets.save_kaggle_api_token(
                            provider_id,
                            existing_kaggle_api_token,
                        )
                    else:
                        credential_secrets.delete_kaggle_api_token(provider_id)
                except Exception:
                    logger.exception(
                        "provider.update_kaggle_token_rollback_failed",
                        provider_id=provider_id,
                    )
            raise

    if (
        not metadata_requested
        and not payload.encrypted_api_key
        and not payload.clear_api_key
        and not payload.encrypted_kaggle_api_token
        and not payload.clear_kaggle_api_token
    ):
        raise HTTPException(status_code = 400, detail = "No fields to update")

    row = providers_db.get_provider(provider_id)
    return _provider_response(row)


@router.put("/{provider_id}/api-key/migrate", response_model = ProviderResponse)
@serialize_provider_config
async def migrate_provider_api_key(
    provider_id: str,
    payload: ProviderCredentialMigration,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Insert a browser legacy key only when this provider has no saved key."""
    require_ui_session(via_api_key)
    if providers_db.get_provider(provider_id) is None:
        raise HTTPException(status_code = 404, detail = "Provider not found")
    api_key = resolve_provider_api_key_or_400(
        None, payload.encrypted_api_key, allow_saved_key = False
    )
    if not api_key:
        raise HTTPException(status_code = 400, detail = "API key cannot be empty")
    credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        credential_secrets.save_provider_api_key_if_absent(provider_id, api_key)
    return _provider_response(providers_db.get_provider(provider_id))


@router.delete("/{provider_id}", status_code = 204)
@serialize_provider_config
async def delete_provider_config(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Idempotently delete a saved provider and its installation credential."""
    require_ui_session(via_api_key)
    await openai_codex_auth.cancel_provider_flows(provider_id)
    credential_secrets.get_or_create_credential_encryption_key()

    async with openai_codex_auth.provider_oauth_write_guard(provider_id):
        with current_credential_write(credential):
            existing_api_key = credential_secrets.get_provider_api_key(provider_id)
            existing_kaggle_api_token = credential_secrets.get_kaggle_api_token(provider_id)
            existing_oauth = credential_secrets.get_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_KIND, provider_id
            )

            existing_oauth_flow = credential_secrets.get_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_FLOW_KIND, provider_id
            )
            credential_secrets.delete_provider_api_key(provider_id)
            credential_secrets.delete_kaggle_api_token(provider_id)
            credential_secrets.delete_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_KIND, provider_id
            )
            credential_secrets.delete_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_FLOW_KIND, provider_id
            )
            try:
                providers_db.delete_provider(provider_id)
            except Exception:
                try:
                    if existing_api_key:
                        credential_secrets.save_provider_api_key(provider_id, existing_api_key)
                    if existing_kaggle_api_token:
                        credential_secrets.save_kaggle_api_token(
                            provider_id,
                            existing_kaggle_api_token,
                        )
                    if existing_oauth:
                        credential_secrets.upsert_secret(
                            credential_secrets.OPENAI_CODEX_OAUTH_KIND,
                            provider_id,
                            existing_oauth,
                        )

                    if existing_oauth_flow:
                        credential_secrets.upsert_secret(
                            credential_secrets.OPENAI_CODEX_OAUTH_FLOW_KIND,
                            provider_id,
                            existing_oauth_flow,
                        )
                except Exception:
                    logger.exception(
                        "provider.delete_credential_rollback_failed", provider_id = provider_id
                    )
                raise
    await kaggle_tpu_manager.detach(provider_id)


def _bind_saved_provider_target(payload):
    """Use the saved provider's endpoint whenever its saved credential may be used."""
    if not payload.provider_id or payload.encrypted_api_key:
        return payload
    config = providers_db.get_provider(payload.provider_id)
    if config is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Provider config not found: {payload.provider_id}",
        )
    if not config["is_enabled"]:
        raise HTTPException(
            status_code = 400,
            detail = f"Provider '{config['display_name']}' is disabled.",
        )
    return payload.model_copy(
        update = {
            "provider_type": config["provider_type"],
            "base_url": config["base_url"],
        }
    )


async def _test_openai_compatible_connection(
    client: ExternalProviderClient,
    model_id: str,
) -> ProviderTestResult:
    """Prefer model discovery, but keep manual-only compatible servers usable."""
    try:
        models = await client.list_models()
    except Exception:
        if not model_id:
            raise
        models = []
    if models:
        return ProviderTestResult(
            success=True,
            message=f"Connected successfully. Found {len(models)} model(s).",
            models_count=len(models),
        )
    if not model_id:
        return ProviderTestResult(
            success=True,
            message="Connected successfully. The server returned no model catalog; enter a model ID manually.",
            models_count=0,
        )
    await client.chat_completion(
        messages=[{"role": "user", "content": "ping"}],
        model=model_id,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
    )
    return ProviderTestResult(
        success=True,
        message="Connected successfully. Chat completions endpoint responded.",
        models_count=0,
    )


# ── Test connectivity ─────────────────────────────────────────────


@router.post("/test", response_model = ProviderTestResult)
async def test_provider(
    payload: ProviderTestRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """
    Test connectivity to an external provider.

    Makes a lightweight GET /models call to verify the API key works. Generic
    custom endpoints use a chat-completions probe because /models is optional.
    An explicit encrypted key takes precedence over the saved provider key.
    """

    payload = _bind_saved_provider_target(payload)
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    api_key = resolve_provider_api_key_or_400(
        payload.provider_id,
        payload.encrypted_api_key,
        allow_saved_key = not via_api_key,
    )

    base_url = payload.base_url or info["base_url"]
    if payload.provider_type in ("custom", "openai_compatible", "kaggle_tpu"):
        if not base_url:
            return ProviderTestResult(
                success = False,
                message = "Connection failed: Base URL is required for custom providers.",
                models_count = None,
            )
    try:
        base_url = validate_provider_base_url(base_url)
    except ValueError as exc:
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {exc}",
            models_count = None,
        )

    client = ExternalProviderClient(
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        timeout = 15.0,
    )

    try:
        if payload.provider_type == "custom":
            model_id = (payload.model_id or "").strip()
            if not model_id:
                return ProviderTestResult(
                    success = False,
                    message = "Connection failed: add a model ID to test custom providers.",
                    models_count = None,
                )
            await client.chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = model_id,
                temperature = 0.0,
                top_p = 1.0,
                max_tokens = 1,
            )
            return ProviderTestResult(
                success = True,
                message = "Connected successfully. Chat completions endpoint responded.",
                models_count = None,
            )
        if payload.provider_type in ("openai_compatible", "kaggle_tpu"):
            model_id = (payload.model_id or "").strip()
            return await _test_openai_compatible_connection(client, model_id)
        if info.get("model_list_mode") == "curated":
            await client.verify_models_endpoint_lightweight()
            return ProviderTestResult(
                success = True,
                message = (
                    "Connected successfully. Full model list is not fetched for this provider — "
                    "use suggestions and manual model IDs in the dialog."
                ),
                models_count = None,
            )
        models = await client.list_models()
        return ProviderTestResult(
            success = True,
            message = f"Connected successfully. Found {len(models)} model(s).",
            models_count = len(models),
        )
    except Exception as exc:
        if payload.provider_type in ("openai_compatible", "kaggle_tpu"):
            logger.error(
                "providers.test_failed",
                provider_type=payload.provider_type,
                error_type=type(exc).__name__,
            )
        else:
            logger.error(
                "providers.test_failed",
                provider_type = payload.provider_type,
                error = str(exc),
                exc_info = True,
            )
        if payload.provider_type in ("openai_compatible", "kaggle_tpu"):
            return ProviderTestResult(
                success=False,
                message=(
                    "Connection failed: Kaggle TPU backend unavailable. Check the session, "
                    "Cloudflare tunnel and generated API key, then use Reconnect."
                    if payload.provider_type == "kaggle_tpu"
                    else "Connection failed: the remote OpenAI-compatible server is unavailable. "
                    "Check the endpoint, API key and model ID."
                ),
                models_count=None,
            )
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {safe_curated_detail(exc)}",
            models_count = None,
        )
    finally:
        await client.close()


# ── Managed Kaggle TPU lifecycle ─────────────────────────────────


def _kaggle_lifecycle_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/{provider_id}/kaggle/status", response_model=KaggleTPULifecycleStatus)
async def kaggle_tpu_status(
    provider_id: str,
    _current_subject: str = Depends(get_current_subject),
):
    try:
        return await kaggle_tpu_manager.status(provider_id)
    except ValueError as exc:
        raise _kaggle_lifecycle_error(exc) from None


@router.post("/{provider_id}/kaggle/start", response_model=KaggleTPULifecycleStatus)
async def start_kaggle_tpu(
    provider_id: str,
    _credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    try:
        return await kaggle_tpu_manager.start(provider_id)
    except (OSError, ValueError) as exc:
        raise _kaggle_lifecycle_error(exc) from None


@router.post("/{provider_id}/kaggle/reconnect", response_model=KaggleTPULifecycleStatus)
async def reconnect_kaggle_tpu(
    provider_id: str,
    _credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    try:
        return await kaggle_tpu_manager.refresh(provider_id)
    except (OSError, ValueError) as exc:
        raise _kaggle_lifecycle_error(exc) from None


@router.post("/{provider_id}/kaggle/stop", response_model=KaggleTPULifecycleStatus)
async def stop_kaggle_tpu(
    provider_id: str,
    _credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    try:
        return await kaggle_tpu_manager.stop(provider_id)
    except (OSError, ValueError) as exc:
        raise _kaggle_lifecycle_error(exc) from None


# ── List models from provider ─────────────────────────────────────


@router.post("/models", response_model = list[ProviderModelInfo])
async def list_provider_models(
    payload: ProviderModelsRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """
    List models available from an external provider.

    An explicit encrypted key takes precedence over the saved provider key.
    """

    payload = _bind_saved_provider_target(payload)
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    api_key = resolve_provider_api_key_or_400(
        payload.provider_id,
        payload.encrypted_api_key,
        allow_saved_key = not via_api_key,
    )

    if info.get("model_list_mode") == "curated":
        return [
            ProviderModelInfo(
                id = m,
                display_name = m,
                context_length = None,
                owned_by = None,
            )
            for m in info.get("default_models", [])
        ]

    base_url = payload.base_url or info["base_url"]
    try:
        base_url = validate_provider_base_url(base_url)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

    client = ExternalProviderClient(
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        timeout = 15.0,
    )

    try:
        models = await client.list_models()
        # Registry model-id filters only apply to the native Gemini base. A
        # custom OAI-compatible proxy returns prefixed IDs the native allowlist
        # would strip, leaving the picker empty; match the host check here so the
        # model list and chat dispatch agree on what counts as "native".
        apply_registry_model_filters = True
        if payload.provider_type == "gemini":
            try:
                from urllib.parse import urlparse as _urlparse
                _host = (_urlparse(base_url).hostname or "").lower()
            except Exception:
                _host = ""
            apply_registry_model_filters = _host == "generativelanguage.googleapis.com"

        if apply_registry_model_filters:
            allow_prefixes = info.get("model_id_allow_prefixes")
            if allow_prefixes is not None:
                prefix_tuple = tuple(str(p) for p in allow_prefixes if str(p))
                if prefix_tuple:
                    models = [m for m in models if m.get("id", "").startswith(prefix_tuple)]
            allowlist = info.get("model_id_allowlist")
            if allowlist is not None:
                models = [m for m in models if allowlist.match(m.get("id", ""))]
            deny_exact = info.get("model_id_deny_exact")
            if deny_exact is not None:
                deny_ids = {str(m) for m in deny_exact if str(m)}
                if deny_ids:
                    models = [m for m in models if m.get("id", "") not in deny_ids]
            denylist = info.get("model_id_denylist")
            if denylist is not None:
                models = [m for m in models if not denylist.search(m.get("id", ""))]
        # Optional cap after filtering to keep large catalogs picker-sized.
        # Unsorted, so "first N matches"; pair with default_models for flagships.
        limit = info.get("model_id_limit")
        if isinstance(limit, int) and limit > 0:
            models = models[:limit]
        return [
            ProviderModelInfo(
                id = m.get("id", ""),
                display_name = m.get("id", ""),
                context_length = m.get("context_length") or m.get("context_window") or m.get("max_model_len"),
                owned_by = m.get("owned_by"),
            )
            for m in models
        ]
    except Exception as exc:
        if payload.provider_type in ("openai_compatible", "kaggle_tpu"):
            logger.error(
                "providers.list_models_failed",
                provider_type=payload.provider_type,
                error_type=type(exc).__name__,
            )
            detail = (
                "Failed to list Kaggle TPU models. Check the session, tunnel and API key."
                if payload.provider_type == "kaggle_tpu"
                else "Failed to list models from the remote OpenAI-compatible server."
            )
            raise HTTPException(status_code=502, detail=detail) from None
        raise log_and_http_error(
            exc,
            502,
            f"Failed to list models from {payload.provider_type}.",
            event = "providers.list_models_failed",
            log = logger,
        )
    finally:
        await client.close()
