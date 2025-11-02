from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel

from ...core.config.settings import Settings
from ..contracts.tokenizer import TokenizerHandle


class ModelTrainConfig(BaseModel):
    model_family: Literal["gpt2", "llama", "qwen"]
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
    cancelled: bool = False

    model_config = {"extra": "forbid", "validate_assignment": True}


class EvalOutcome(BaseModel):
    loss: float
    perplexity: float

    model_config = {"extra": "forbid", "validate_assignment": True}


class PreparedModel(Protocol):
    """Opaque prepared model handle used by backends."""

    ...


class ModelArtifact(BaseModel):
    out_dir: str

    model_config = {"extra": "forbid", "validate_assignment": True}


class ModelBackend(Protocol):
    def name(self: ModelBackend) -> str: ...
    def prepare(
        self: ModelBackend, cfg: ModelTrainConfig, settings: Settings, *, tokenizer: TokenizerHandle
    ) -> PreparedModel: ...
    def save(self: ModelBackend, prepared: PreparedModel, out_dir: str) -> ModelArtifact: ...
    def load(
        self: ModelBackend, artifact_path: str, settings: Settings, *, tokenizer: TokenizerHandle
    ) -> PreparedModel: ...
    # Training & evaluation
    def train(
        self: ModelBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedModel,
    ) -> TrainOutcome: ...
    def evaluate(
        self: ModelBackend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome: ...
