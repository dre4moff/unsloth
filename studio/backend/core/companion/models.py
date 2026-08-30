from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


PROTOCOL_VERSION = 1
SERVICE_TYPE = "_unsloth-cp._tcp.local."


class ConnectionMode(str, Enum):
    automatic_best = "automatic_best"
    multiple = "multiple"


class CompanionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    mode: ConnectionMode = ConnectionMode.automatic_best
    selectedDeviceIDs: list[UUID] = Field(default_factory=list)


class CapabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocolVersion: int
    deviceID: UUID
    deviceName: str
    runtimeVersion: str
    taskKinds: set[str]
    selectedModelID: str | None
    supportsVision: bool
    supportsAudio: bool
    contextSize: int
    maxRawMediaBytes: int


class DeviceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deviceID: UUID
    name: str
    enabled: bool = True
    connected: bool = False
    authenticated: bool = False
    battery: float = 0
    lowPowerMode: bool = False
    thermalState: int = 0
    state: str = "offline"
    latencyMS: float = 0
    queueDepth: int = 0
    storageFreeBytes: int = 0
    score: float = 0
    capabilities: CapabilityReport | None = None
    lastSeen: datetime | None = None


class PairedDevice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deviceID: UUID
    name: str
    signingPublicKey: str
    pairedAt: datetime
    enabled: bool = True


class PendingPairing(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    pairingID: UUID
    deviceID: UUID
    deviceName: str
    signingPublicKey: str
    code: str
    createdAt: datetime
    phoneConfirmed: bool = False
    desktopConfirmed: bool = False


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocolVersion: int = PROTOCOL_VERSION
    messageID: UUID = Field(default_factory=uuid4)
    sentAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0))
    type: str
    payload: dict[str, Any]


class CompanionTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    taskID: UUID
    parentTaskID: UUID | None = None
    shardIndex: int | None = None
    shardCount: int | None = None
    idempotencyKey: str = Field(min_length=16, max_length=128)
    kind: Literal[
        "subagent", "classification", "summary", "context_compression", "extraction",
        "verification", "reranking", "lightweight_planning", "vision", "ocr",
        "video_summary", "audio_transcription", "audio_analysis", "dsp",
    ]
    priority: int = Field(ge=0, le=100)
    timeoutSeconds: float = Field(gt=0, le=3600)
    leaseSeconds: float = Field(default=30, ge=10, le=120)
    mediaPolicy: Literal["semantic_only", "derived_media", "raw_media"] = "semantic_only"
    input: dict[str, Any]
    resultSchema: dict[str, Any] = Field(default_factory=dict)


class CompanionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    taskID: UUID
    result: dict[str, Any]
    sha256: str
    durationMS: int
    tokensGenerated: int


class ActiveCompanionTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    taskID: UUID
    deviceID: UUID
    deviceName: str
    kind: str
    startedAt: datetime


class CompanionStatus(BaseModel):
    settings: CompanionSettings
    devices: list[DeviceStatus]
    pendingPairings: list[PendingPairing]
    activeTasks: list[ActiveCompanionTask]
    listenerPort: int | None
    certificateSHA256: str | None
