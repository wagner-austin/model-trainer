from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    corpus_path: str
    holdout_fraction: float = 0.01

    model_config = {"extra": "forbid", "validate_assignment": True}


class DatasetBuilder(Protocol):
    def split(self: DatasetBuilder, cfg: DatasetConfig) -> tuple[list[str], list[str]]: ...
