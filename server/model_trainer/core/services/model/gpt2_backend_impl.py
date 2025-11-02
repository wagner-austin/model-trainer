from __future__ import annotations

from collections.abc import Callable

from ...config.settings import Settings
from ...contracts.dataset import DatasetBuilder
from ...contracts.model import (
    EvalOutcome,
    ModelArtifact,
    ModelBackend,
    ModelTrainConfig,
    PreparedModel,
    TrainOutcome,
)
from ...contracts.tokenizer import TokenizerHandle
from .backends.gpt2 import (
    GPT2Prepared,
    GPT2TrainConfig,
    evaluate_gpt2,
    load_prepared_gpt2_from_handle,
    prepare_gpt2_with_handle,
    save_prepared_gpt2,
    train_prepared_gpt2,
)


class GPT2BackendImpl(ModelBackend):
    def __init__(self: GPT2BackendImpl, dataset_builder: DatasetBuilder) -> None:
        self._ds = dataset_builder

    def name(self: GPT2BackendImpl) -> str:
        return "gpt2"

    def prepare(
        self: GPT2BackendImpl,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedModel:
        return prepare_gpt2_with_handle(
            tokenizer,
            GPT2TrainConfig(
                model_family="gpt2",
                model_size=cfg.model_size,
                max_seq_len=cfg.max_seq_len,
                num_epochs=cfg.num_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                tokenizer_id=cfg.tokenizer_id,
                corpus_path=cfg.corpus_path,
            ),
        )

    def save(self: GPT2BackendImpl, prepared: PreparedModel, out_dir: str) -> ModelArtifact:
        if isinstance(prepared, GPT2Prepared):
            save_prepared_gpt2(prepared, out_dir)
            return ModelArtifact(out_dir=out_dir)
        raise TypeError("Unsupported prepared model type for GPT-2 backend")

    def load(
        self: GPT2BackendImpl,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedModel:
        return load_prepared_gpt2_from_handle(artifact_path, tokenizer)

    def train(
        self: GPT2BackendImpl,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedModel,
    ) -> TrainOutcome:
        if not isinstance(prepared, GPT2Prepared):
            raise TypeError("Unsupported prepared model type for GPT-2 backend")
        out = train_prepared_gpt2(
            prepared,
            GPT2TrainConfig(
                model_family="gpt2",
                model_size=cfg.model_size,
                max_seq_len=cfg.max_seq_len,
                num_epochs=cfg.num_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                tokenizer_id=cfg.tokenizer_id,
                corpus_path=cfg.corpus_path,
            ),
            settings,
            run_id=run_id,
            redis_hb=heartbeat,
            cancelled=cancelled,
        )
        return TrainOutcome(
            loss=out.loss,
            perplexity=out.perplexity,
            steps=out.steps,
            out_dir=out.out_dir,
            cancelled=out.cancelled,
        )

    def evaluate(
        self: GPT2BackendImpl, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome:
        res = evaluate_gpt2(
            run_id=run_id,
            cfg=GPT2TrainConfig(
                model_family="gpt2",
                model_size=cfg.model_size,
                max_seq_len=cfg.max_seq_len,
                num_epochs=cfg.num_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                tokenizer_id=cfg.tokenizer_id,
                corpus_path=cfg.corpus_path,
            ),
            settings=settings,
            dataset_builder=self._ds,
        )
        return EvalOutcome(loss=res.loss, perplexity=res.perplexity)
