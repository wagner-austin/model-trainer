from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ArtifactPointer(BaseModel):
    storage: Literal["data-bank"]
    file_id: str

    model_config = {"extra": "forbid", "validate_assignment": True}
