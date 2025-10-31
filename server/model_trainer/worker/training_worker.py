from __future__ import annotations

import logging
import time
from typing import TypedDict

import redis

HEARTBEAT_KEY_PREFIX = "runs:hb:"
STATUS_KEY_PREFIX = "runs:status:"
EVAL_KEY_PREFIX = "runs:eval:"


class TrainJobPayload(TypedDict):
    run_id: str
    request: dict[str, str | int | float]


class EvalJobPayload(TypedDict):
    run_id: str
    split: str
    path_override: str | None


def _redis_client() -> redis.Redis[str]:
    # Worker environment must provide REDIS_URL
    import os

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def process_train_job(payload: TrainJobPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    run_id = payload["run_id"]
    r.set(f"{STATUS_KEY_PREFIX}{run_id}", "running")

    # Placeholder: simulate work and heartbeats
    for _ in range(3):
        r.set(f"{HEARTBEAT_KEY_PREFIX}{run_id}", str(time.time()))
        time.sleep(1)

    r.set(f"{STATUS_KEY_PREFIX}{run_id}", "completed")
    log.info("Training job completed run_id=%s", run_id)


class _EvalCacheModel(BaseModel):
    status: str
    split: str
    loss: float | None = None
    ppl: float | None = None
    artifact: str | None = None


def process_eval_job(payload: EvalJobPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    run_id = payload["run_id"]
    split = payload["split"]
    running = _EvalCacheModel(status="running", split=split)
    r.set(f"{EVAL_KEY_PREFIX}{run_id}", running.model_dump_json())
    time.sleep(1)
    # Placeholder metrics
    out = _EvalCacheModel(status="completed", split=split, loss=10.0, ppl=22.0)
    r.set(f"{EVAL_KEY_PREFIX}{run_id}", out.model_dump_json())
    log.info("Eval job completed run_id=%s split=%s", run_id, split)
from pydantic import BaseModel

