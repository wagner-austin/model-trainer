from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel


class HealthzResponse(BaseModel):
    model_config: ClassVar[dict[str, object]] = {"extra": "forbid", "validate_assignment": True}

    status: Literal["ok"]


class ReadyzResponse(BaseModel):
    model_config: ClassVar[dict[str, object]] = {"extra": "forbid", "validate_assignment": True}

    status: Literal["ready", "degraded"]
    reason: Literal["no-worker", "redis no-pong", "redis error"] | None = None
