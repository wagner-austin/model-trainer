from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ArtifactListResponse(BaseModel):
    kind: Literal["tokenizers", "models"]
    item_id: str
    files: list[str]
