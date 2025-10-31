from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel

from ...core.config.settings import Settings


class ModelTrainConfig(BaseModel):
    model_family: Literal["gpt2"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str
    corpus_path: str

    model_config = {"extra": "forbid", "validate_assignment": True}


class TrainOutcome(BaseModel):
    loss: float
    perplexity: float
    steps: int
    out_dir: str

    model_config = {"extra": "forbid", "validate_assignment": True}


class EvalOutcome(BaseModel):
    loss: float
    perplexity: float

    model_config = {"extra": "forbid", "validate_assignment": True}


class ModelBackend(Protocol):
    def name(self: ModelBackend) -> str: ...
    def train(
        self: ModelBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
    ) -> TrainOutcome: ...
    def evaluate(
        self: ModelBackend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome: ...
