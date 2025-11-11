from __future__ import annotations

import json
from typing import Literal, NotRequired, TypedDict


class StartedV1(TypedDict):
    type: Literal["trainer.train.started.v1"]
    request_id: str
    run_id: str
    user_id: int
    model_family: str
    model_size: str
    total_epochs: int
    queue: str
    # Optional rich runtime context
    cpu_cores: NotRequired[int]
    memory_mb: NotRequired[int]
    optimal_threads: NotRequired[int]
    optimal_workers: NotRequired[int]
    batch_size: NotRequired[int]
    learning_rate: NotRequired[float]


class ProgressV1(TypedDict):
    type: Literal["trainer.train.progress.v1"]
    request_id: str
    run_id: str
    user_id: int
    epoch: int
    total_epochs: int
    step: int
    loss: float
    # Optional throughput/memory metrics
    samples_per_sec: NotRequired[float]
    main_rss_mb: NotRequired[int]
    workers_rss_mb: NotRequired[int]
    worker_count: NotRequired[int]
    cgroup_usage_mb: NotRequired[int]
    cgroup_limit_mb: NotRequired[int]
    cgroup_pct: NotRequired[float]


class CompletedV1(TypedDict):
    type: Literal["trainer.train.completed.v1"]
    request_id: str
    run_id: str
    user_id: int
    loss: float
    perplexity: float
    artifact_path: str


class FailedV1(TypedDict):
    type: Literal["trainer.train.failed.v1"]
    request_id: str
    run_id: str
    user_id: int
    error_kind: Literal["user", "system"]
    message: str
    status: Literal["failed", "canceled"]


Event = StartedV1 | ProgressV1 | CompletedV1 | FailedV1


def encode_event(ev: Event) -> str:
    return json.dumps(ev, separators=(",", ":"))
