from __future__ import annotations

import redis
from dataclasses import dataclass

from ..config.settings import Settings
from .queue.rq_adapter import RQEnqueuer, RQSettings
from ...orchestrators.training_orchestrator import TrainingOrchestrator
from ...orchestrators.tokenizer_orchestrator import TokenizerOrchestrator


@dataclass(frozen=True)
class ServiceContainer:
    settings: Settings
    redis: redis.Redis[str]
    rq_enqueuer: RQEnqueuer
    training_orchestrator: TrainingOrchestrator
    tokenizer_orchestrator: TokenizerOrchestrator

    @classmethod
    def from_settings(cls, settings: Settings) -> ServiceContainer:
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
        return cls(
            settings=settings,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=training,
            tokenizer_orchestrator=tokenizer,
        )

