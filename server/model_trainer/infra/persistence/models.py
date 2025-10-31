from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EvalCache(BaseModel):
    status: Literal["queued", "running", "completed", "failed"]
    split: str
    loss: float | None = None
    ppl: float | None = None
    artifact: str | None = None

