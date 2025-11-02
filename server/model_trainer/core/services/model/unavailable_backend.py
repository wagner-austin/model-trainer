from __future__ import annotations

from collections.abc import Callable

from ...config.settings import Settings
from ...contracts.model import (
    EvalOutcome,
    ModelArtifact,
    ModelBackend,
    ModelTrainConfig,
    PreparedModel,
    TrainOutcome,
)
from ...contracts.tokenizer import TokenizerHandle
from ...errors.base import AppError, ErrorCode


class UnavailableBackend(ModelBackend):
    def __init__(self: UnavailableBackend, name: str) -> None:
        self._name = name

    def name(self: UnavailableBackend) -> str:
        return self._name

    def prepare(
        self: UnavailableBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedModel:
        raise AppError(ErrorCode.CONFIG_INVALID, f"model backend unavailable: {self._name}")

    def save(self: UnavailableBackend, prepared: PreparedModel, out_dir: str) -> ModelArtifact:
        raise AppError(ErrorCode.CONFIG_INVALID, f"model backend unavailable: {self._name}")

    def load(
        self: UnavailableBackend,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedModel:
        raise AppError(ErrorCode.CONFIG_INVALID, f"model backend unavailable: {self._name}")

    def train(
        self: UnavailableBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedModel,
    ) -> TrainOutcome:
        raise AppError(ErrorCode.CONFIG_INVALID, f"model backend unavailable: {self._name}")

    def evaluate(
        self: UnavailableBackend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome:
        raise AppError(ErrorCode.CONFIG_INVALID, f"model backend unavailable: {self._name}")
