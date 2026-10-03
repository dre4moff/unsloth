from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth.authentication import get_current_subject, authenticated_via_api_key
from core.companion import companion_manager
from core.companion.models import CompanionSettings, CompanionStatus
from core.companion.backburner import backburner_manager
import asyncio
from typing import Literal


router = APIRouter()


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def nonempty_trimmed_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Device name cannot be empty")
        return value


class EnabledRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


@router.get("/status", response_model=CompanionStatus)
async def status(_: str = Depends(get_current_subject)) -> CompanionStatus:
    return companion_manager.status()


@router.put("/settings", response_model=CompanionStatus)
async def update_settings(value: CompanionSettings, _: str = Depends(get_current_subject)) -> CompanionStatus:
    try:
        return await companion_manager.update_settings(value)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class AccelerationModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["agent", "speed"]


class AccelerationPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modelPath: str = Field(min_length=1, max_length=4096)
    draftPath: str = Field(min_length=1, max_length=4096)


@router.get("/acceleration/status")
async def acceleration_status(_: str = Depends(get_current_subject)):
    from routes.inference import get_llama_cpp_backend
    value = await asyncio.to_thread(backburner_manager.status)
    value["modelPath"] = get_llama_cpp_backend().gguf_path
    return value


@router.put("/acceleration/mode")
async def acceleration_mode(value: AccelerationModeRequest, _: str = Depends(get_current_subject)):
    from routes.inference import get_llama_cpp_backend
    try:
        return await backburner_manager.select_mode(value.mode, get_llama_cpp_backend(), companion_manager)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/acceleration/prepare")
async def acceleration_prepare(value: AccelerationPrepareRequest, _: str = Depends(get_current_subject), via_api_key: bool = Depends(authenticated_via_api_key)):
    from routes.provider_credentials import require_ui_session
    require_ui_session(via_api_key)
    try:
        return await backburner_manager.prepare(value.modelPath, value.draftPath, companion_manager)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/pairings/{pairing_id}/confirm", status_code=204)
async def confirm_pairing(pairing_id: UUID, _: str = Depends(get_current_subject)) -> None:
    try: await companion_manager.confirm_pairing(pairing_id)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/pairings/{pairing_id}/reject", status_code=204)
async def reject_pairing(pairing_id: UUID, _: str = Depends(get_current_subject)) -> None:
    try: await companion_manager.reject_pairing(pairing_id)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/devices/{device_id}", status_code=204)
async def revoke(device_id: UUID, _: str = Depends(get_current_subject)) -> None:
    await companion_manager.revoke(device_id)


@router.put("/devices/{device_id}/name", status_code=204)
async def rename(device_id: UUID, value: RenameRequest, _: str = Depends(get_current_subject)) -> None:
    try: await companion_manager.rename(device_id, value.name)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/devices/{device_id}/enabled", status_code=204)
async def set_enabled(device_id: UUID, value: EnabledRequest, _: str = Depends(get_current_subject)) -> None:
    try: await companion_manager.set_device_enabled(device_id, value.enabled)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/cancel", status_code=204)
async def cancel_task(task_id: UUID, _: str = Depends(get_current_subject)) -> None:
    await companion_manager.cancel(task_id, explicit_user=True)
