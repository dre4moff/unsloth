from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth.authentication import get_current_subject
from core.companion import companion_manager
from core.companion.models import CompanionSettings, CompanionStatus


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
    return await companion_manager.update_settings(value)


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
