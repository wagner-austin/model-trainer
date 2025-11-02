from __future__ import annotations

import shutil
from dataclasses import dataclass

import redis

from ...orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from ...orchestrators.training_orchestrator import TrainingOrchestrator
from ..config.settings import Settings
from ..contracts.dataset import DatasetBuilder
from ..contracts.tokenizer import TokenizerBackend
from ..logging.service import LoggingService
from ..services.dataset.local_text_builder import LocalTextDatasetBuilder
from ..services.registries import ModelRegistry, TokenizerRegistry
from .model.gpt2_backend_impl import GPT2BackendImpl
from .model.unavailable_backend import UnavailableBackend
from .queue.rq_adapter import RQEnqueuer, RQSettings
from .tokenizer.bpe_backend import BPEBackend


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
        enq = _create_enqueuer(settings)
        dataset_builder = LocalTextDatasetBuilder()

        # Registries (minimal initial backends)
        model_registry = _create_model_registry(dataset_builder)
        training = TrainingOrchestrator(
            settings=settings, redis_client=r, enqueuer=enq, model_registry=model_registry
        )
        tokenizer = TokenizerOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
        tokenizer_registry = _create_tokenizer_registry()
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


def _create_model_registry(dataset_builder: DatasetBuilder) -> ModelRegistry:
    return ModelRegistry(
        backends={
            "gpt2": GPT2BackendImpl(dataset_builder),
            "llama": UnavailableBackend("llama"),
            "qwen": UnavailableBackend("qwen"),
        }
    )


def _create_enqueuer(settings: Settings) -> RQEnqueuer:
    rq_cfg = RQSettings(
        queue_name=settings.rq.queue_name,
        job_timeout_sec=settings.rq.job_timeout_sec,
        result_ttl_sec=settings.rq.result_ttl_sec,
        failure_ttl_sec=settings.rq.failure_ttl_sec,
        retry_max=settings.rq.retry_max,
        retry_intervals=[int(x) for x in settings.rq.retry_intervals_sec.split(",") if x],
    )
    return RQEnqueuer(redis_url=settings.redis.url, settings=rq_cfg)


def _create_tokenizer_registry() -> TokenizerRegistry:
    tok_backends: dict[str, TokenizerBackend] = {"bpe": BPEBackend()}
    if all(shutil.which(x) is not None for x in ("spm_train", "spm_encode", "spm_decode")):
        from .tokenizer.spm_backend import SentencePieceBackend

        tok_backends["sentencepiece"] = SentencePieceBackend()
    return TokenizerRegistry(backends=tok_backends)
