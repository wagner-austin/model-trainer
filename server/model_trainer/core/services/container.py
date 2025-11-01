from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass

import redis

from ...orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from ...orchestrators.training_orchestrator import TrainingOrchestrator
from ..config.settings import Settings
from ..contracts.dataset import DatasetBuilder
from ..contracts.model import EvalOutcome, ModelBackend, ModelTrainConfig, TrainOutcome
from ..contracts.tokenizer import TokenizerBackend
from ..logging.service import LoggingService
from ..services.dataset.local_text_builder import LocalTextDatasetBuilder
from ..services.registries import ModelRegistry, TokenizerRegistry
from .queue.rq_adapter import RQEnqueuer, RQSettings
from .tokenizer.bpe_backend import BPEBackend
from .training.gpt2_backend import GPT2TrainConfig, evaluate_gpt2, train_gpt2


@dataclass
class ServiceContainer:
    settings: Settings
    redis: redis.Redis[str]
    rq_enqueuer: RQEnqueuer
    training_orchestrator: TrainingOrchestrator
    tokenizer_orchestrator: TokenizerOrchestrator
    model_registry: ModelRegistry
    tokenizer_registry: TokenizerRegistry
    logging: LoggingService
    dataset_builder: DatasetBuilder

    @classmethod
    def from_settings(cls: type[ServiceContainer], settings: Settings) -> ServiceContainer:
        r = redis.from_url(settings.redis.url, decode_responses=True)
        rq_cfg = RQSettings(
            queue_name=settings.rq.queue_name,
            job_timeout_sec=settings.rq.job_timeout_sec,
            result_ttl_sec=settings.rq.result_ttl_sec,
            failure_ttl_sec=settings.rq.failure_ttl_sec,
            retry_max=settings.rq.retry_max,
            retry_intervals=[int(x) for x in settings.rq.retry_intervals_sec.split(",") if x],
        )
        enq = RQEnqueuer(redis_url=settings.redis.url, settings=rq_cfg)
        training = TrainingOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
        tokenizer = TokenizerOrchestrator(settings=settings, redis_client=r, enqueuer=enq)

        dataset_builder = LocalTextDatasetBuilder()

        # Registries (minimal initial backends)
        class _GPT2Backend(ModelBackend):
            def name(self: _GPT2Backend) -> str:
                return "gpt2"

            def train(
                self: _GPT2Backend,
                cfg: ModelTrainConfig,
                settings: Settings,
                *,
                run_id: str,
                heartbeat: Callable[[float], None],
                cancelled: Callable[[], bool],
            ) -> TrainOutcome:
                out = train_gpt2(
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
                    dataset_builder=dataset_builder,
                )
                return TrainOutcome(
                    loss=out.loss, perplexity=out.perplexity, steps=out.steps, out_dir=out.out_dir
                )

            def evaluate(
                self: _GPT2Backend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
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
                    dataset_builder=dataset_builder,
                )
                return EvalOutcome(loss=res.loss, perplexity=res.perplexity)

        model_registry = ModelRegistry(backends={"gpt2": _GPT2Backend()})
        tok_backends: dict[str, TokenizerBackend] = {"bpe": BPEBackend()}
        if all(shutil.which(x) is not None for x in ("spm_train", "spm_encode", "spm_decode")):
            from .tokenizer.spm_backend import SentencePieceBackend

            tok_backends["sentencepiece"] = SentencePieceBackend()
        tokenizer_registry = TokenizerRegistry(backends=tok_backends)
        logging_service = LoggingService.create()
        return cls(
            settings=settings,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=training,
            tokenizer_orchestrator=tokenizer,
            model_registry=model_registry,
            tokenizer_registry=tokenizer_registry,
            logging=logging_service,
            dataset_builder=dataset_builder,
        )
