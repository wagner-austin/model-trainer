from __future__ import annotations

from dataclasses import dataclass

import redis

from ...contracts.queue import (
    EvalJobPayload,
    TokenizerTrainPayload,
    TrainJobPayload,
)


@dataclass
class RQSettings:
    queue_name: str
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int
    retry_max: int
    retry_intervals: list[int]


@dataclass
class RQEnqueuer:
    redis_url: str
    settings: RQSettings

    def enqueue_train(self: RQEnqueuer, payload: TrainJobPayload) -> str:
        from rq import Queue, Retry

        conn: redis.Redis[str] = redis.from_url(self.redis_url, decode_responses=True)
        q = Queue(self.settings.queue_name, connection=conn)
        retry = Retry(max=self.settings.retry_max, interval=self.settings.retry_intervals)
        payload_dict: dict[str, object] = dict(payload)
        job = q.enqueue(
            "model_trainer.worker.training_worker.process_train_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"train:{payload['run_id']}",
        )
        return job.get_id()

    def enqueue_eval(self: RQEnqueuer, payload: EvalJobPayload) -> str:
        from rq import Queue, Retry

        conn: redis.Redis[str] = redis.from_url(self.redis_url, decode_responses=True)
        q = Queue(self.settings.queue_name, connection=conn)
        retry = Retry(max=self.settings.retry_max, interval=self.settings.retry_intervals)
        payload_dict: dict[str, object] = dict(payload)
        job = q.enqueue(
            "model_trainer.worker.training_worker.process_eval_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"eval:{payload['run_id']}:{payload['split']}",
        )
        return job.get_id()

    def enqueue_tokenizer(self: RQEnqueuer, payload: TokenizerTrainPayload) -> str:
        from rq import Queue, Retry

        conn: redis.Redis[str] = redis.from_url(self.redis_url, decode_responses=True)
        q = Queue(self.settings.queue_name, connection=conn)
        retry = Retry(max=self.settings.retry_max, interval=self.settings.retry_intervals)
        payload_dict: dict[str, object] = dict(payload)
        job = q.enqueue(
            "model_trainer.worker.tokenizer_worker.process_tokenizer_train_job",
            payload_dict,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"tokenizer:{payload['tokenizer_id']}",
        )
        return job.get_id()
