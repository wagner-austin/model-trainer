from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import redis


class TrainRequestPayload(TypedDict):
    model_family: Literal["gpt2", "llama", "qwen"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    corpus_path: str
    tokenizer_id: str


class TrainJobPayload(TypedDict):
    run_id: str
    request: TrainRequestPayload


class EvalJobPayload(TypedDict):
    run_id: str
    split: str
    path_override: str | None


@dataclass(frozen=True)
class RQSettings:
    queue_name: str
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int
    retry_max: int
    retry_intervals: list[int]


@dataclass(frozen=True)
class RQEnqueuer:
    redis_url: str
    settings: RQSettings

    def enqueue_train(self, payload: TrainJobPayload) -> str:
        from rq import Queue
        from rq.retry import Retry

        conn: redis.Redis[str] = redis.from_url(self.redis_url, decode_responses=True)
        q = Queue(self.settings.queue_name, connection=conn)
        retry = Retry(max=self.settings.retry_max, interval=self.settings.retry_intervals)
        job = q.enqueue(
            "model_trainer.worker.training_worker.process_train_job",
            payload,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"train:{payload['run_id']}",
        )
        return job.get_id()

    def enqueue_eval(self, payload: EvalJobPayload) -> str:
        from rq import Queue
        from rq.retry import Retry

        conn: redis.Redis[str] = redis.from_url(self.redis_url, decode_responses=True)
        q = Queue(self.settings.queue_name, connection=conn)
        retry = Retry(max=self.settings.retry_max, interval=self.settings.retry_intervals)
        job = q.enqueue(
            "model_trainer.worker.training_worker.process_eval_job",
            payload,
            job_timeout=self.settings.job_timeout_sec,
            result_ttl=self.settings.result_ttl_sec,
            failure_ttl=self.settings.failure_ttl_sec,
            retry=retry,
            description=f"eval:{payload['run_id']}:{payload['split']}",
        )
        return job.get_id()
